from __future__ import annotations

import base64
import binascii
import inspect
import json
import os
import re
import tempfile
import types
import warnings
from copy import deepcopy
from io import BytesIO
from os import getenv
from turtle import pu
from typing import TYPE_CHECKING, Any, Generator, Literal, TypeAlias, Unpack
from urllib.parse import urlparse

import openai_stt as stt
import simpleaudio as sa
from openai import OpenAI
from openai.resources.vector_stores.vector_stores import VectorStores
from openai.types.conversations.conversation import Conversation
from openai.types.shared_params import Reasoning, ResponsesModel
from openai.types.vector_store import VectorStore
from PIL import Image
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


class Assistant:
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
        """
        Args:
            api_key (str | None): The API key to use for OpenAI API requests.
            model (ResponsesModel): The model to use for OpenAI API requests.
            system_prompt (str, optional): The system prompt to use for OpenAI API requests. Defaults to "".
            default_conversation (Conversation | bool, optional): The default conversation to use for OpenAI API requests. Defaults to True.
            temperature (float | None, optional): The temperature to use for OpenAI API requests. Defaults to None.
            reasoning_effort (Literal["minimal", "low", "medium", "high"], optional): The reasoning effort to use for OpenAI API requests. Defaults to "medium".
            summary_length (Literal["auto", "concise", "detailed"], optional): The summary length to use for OpenAI API requests. Defaults to "auto".

        Returns:
            Assistant: An instance of the Assistant class.

        Raises:
            ValueError: If no API key is provided.

        Examples:
            bot = Assistant(api_key=None, model="gpt-4o", system_prompt="You are helpful.")


        """

        self.model = model
        if not api_key:
            if not getenv("OPENAI_API_KEY"):
                raise ValueError("No API key provided.")
            else:
                self.api_key = str(getenv("OPENAI_API_KEY"))
        else:
            self.api_key = api_key

        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.summary_length = summary_length
        if reasoning_effort and summary_length:
            self.reasoning = Reasoning(effort=reasoning_effort, summary=summary_length)

        else:
            self.reasoning = None

        if default_conversation is True:
            self.conversation = self.client.conversations.create()
            self.conversation_id = self.conversation.id  # type: ignore
        else:
            self.conversation = None
            self.conversation_id = None


    def _convert_filepath_to_vector(
        self, list_of_files: list[str]
    ) -> tuple[VectorStore, VectorStore, VectorStores]:
        if not isinstance(list_of_files, list) or len(list_of_files) == 0:
            raise ValueError("list_of_files must be a non-empty list of file paths.")
        for filepath in list_of_files:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

        vector_store_create = self.client.vector_stores.create(name="vector_store")
        vector_store = self.client.vector_stores.retrieve(vector_store_create.id)
        vector = self.client.vector_stores
        for filepath in list_of_files:
            with open(filepath, "rb") as f:
                self.client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store_create.id, file=f
                )
        return vector_store_create, vector_store, vector

    def openai_function(self, func: types.FunctionType) -> dict:
        """
        Decorator for OpenAI functions.

        Args:
            func (types.FunctionType): The function to decorate.

        Returns:
            dict: The OpenAI function dictionary.
        """
        if not isinstance(func, types.FunctionType):
            raise TypeError("Expected a plain function (types.FunctionType)")

        doc = inspect.getdoc(func) or ""

        def extract_block(name: str) -> dict:
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

        schema = {
            "type": "function",
            "name": func.__name__,
            "description": description or func.__doc__.strip().split("\n")[0],  # type: ignore
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        func.schema = schema
        return func  # type: ignore

    def _text_stream_generator(
        self,
        params_for_response: dict[str, Any],
        tool_map: dict[str, types.FunctionType],
        *,
        store: bool,
        file_search: bool,
        vector: tuple[VectorStore, VectorStore, VectorStores] | None,
    ) -> Generator[str, Any, None]:
        """Stream response text while resolving custom tool calls."""

        params = deepcopy(params_for_response)
        base_followup = {
            k: deepcopy(v) if isinstance(v, (dict, list)) else v
            for k, v in params.items()
            if k not in {"input", "stream"}
        }

        final_response = None

        while True:
            with self.client.responses.stream(**params) as streamer:
                for event in streamer:
                    if event.type == "response.output_text.delta":
                        yield event.delta

                final_response = streamer.get_final_response()

            tool_calls = self._extract_function_calls(final_response)
            if not tool_calls:
                break

            tool_outputs = self._invoke_custom_tools(tool_calls, tool_map)
            if not tool_outputs:
                break
            params = self._build_followup_params(
                base_followup,
                final_response,
                tool_outputs,
                stream=True,
            )

        if final_response and store and final_response.conversation:
            self.conversation = final_response.conversation

        if file_search and vector:
            vector[2].delete(vector[0].id)

        yield "done"

    def _normalize_tool_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if "function" in schema:
            return schema

        if schema.get("type") != "function":
            raise ValueError("Custom tool schema must describe a function.")

        return {
            "type": "function",
            "function": {k: v for k, v in schema.items() if k != "type"},
        }

    def _extract_function_calls(self, response: Any) -> list[Any]:
        calls: list[Any] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                calls.append(item)
        return calls

    def _invoke_custom_tools(
        self,
        tool_calls: list[Any],
        tool_map: dict[str, types.FunctionType],
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []

        for call in tool_calls:
            tool_name = getattr(call, "name", None)
            if not tool_name or tool_name not in tool_map:
                raise ValueError(f"No custom tool registered with name '{tool_name}'.")

            raw_arguments = getattr(call, "arguments", "") or "{}"
            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed_arguments = raw_arguments

            tool = tool_map[tool_name]

            if isinstance(parsed_arguments, dict):
                result = tool(**parsed_arguments)
            elif isinstance(parsed_arguments, list):
                result = tool(*parsed_arguments)
            else:
                result = tool(parsed_arguments)

            if result is None:
                output_text = ""
            elif isinstance(result, (dict, list)):
                output_text = json.dumps(result)
            else:
                output_text = str(result)

            call_id = getattr(call, "call_id", None) or getattr(call, "id", None)
            if not call_id:
                raise ValueError("Tool call did not include an identifier.")

            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )

        return outputs

    def _build_followup_params(
        self,
        base_params: dict[str, Any],
        response: Any,
        tool_outputs: list[dict[str, Any]],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        params = {
            k: deepcopy(v) if isinstance(v, (dict, list)) else v
            for k, v in base_params.items()
        }

        params["input"] = tool_outputs
        params["previous_response_id"] = getattr(response, "id", None)

        conversation = getattr(response, "conversation", None)
        if conversation and getattr(conversation, "id", None):
            params["conversation"] = conversation.id

        if stream:
            params["stream"] = True
        else:
            params.pop("stream", None)

        return {k: v for k, v in params.items() if v not in ({}, None)}

    def _resolve_function_calls(
        self,
        response: Any,
        initial_params: dict[str, Any],
        tool_map: dict[str, types.FunctionType],
    ) -> Any:
        if not tool_map:
            return response

        base_followup = {
            k: deepcopy(v) if isinstance(v, (dict, list)) else v
            for k, v in initial_params.items()
            if k != "input"
        }

        current_response = response

        while True:
            tool_calls = self._extract_function_calls(current_response)
            if not tool_calls:
                break

            tool_outputs = self._invoke_custom_tools(tool_calls, tool_map)
            if not tool_outputs:
                break
            followup_params = self._build_followup_params(
                base_followup,
                current_response,
                tool_outputs,
                stream=False,
            )

            current_response = self.client.responses.create(**followup_params)

        return current_response

    def chat(
        self,
        input: str,
        conv_id: str | None | Conversation | bool = True,
        images: list["Openai_Images"] = [],
        max_output_tokens: int | None = None,
        store: bool = False,
        web_search: bool = False,
        code_interpreter: bool = False,
        file_search: list[str] = [],
        file_search_max_searches: int | None = None,
        tools_required: Literal["none", "auto", "required"] = "auto",
        custom_tools: list[types.FunctionType] = [],
        return_full_response: bool = False,
        valid_json: dict = {},
        stream: bool = False,
        text_stream: bool = False,
    ) -> str | Generator[str, Any, None]:
        """
        Description:
            This is the chat function

        Args:
            input: The input text.
            conv_id: The conversation ID. Defaults to True. Put the conversation ID here. If you want to create a new conversation, put True.
            max_output_tokens: The maximum output tokens. Defaults to None.
            store: Whether to store the conversation. Defaults to False.
            web_search: Whether to use web search.  Defaults to False.
            code_interpreter: Whether to use code interpreter.  Defaults to False.
            file_search: The file search. Defaults to [].
            tools_required: The tools required. Defaults to "auto".
            custom_tools: The custom tools. Defaults to [].
            file_search_max_searches: The if file search max searches. Defaults to None.
            return_full_response: Whether to return the full response. Defaults to False.
            valid_json: The valid json. Defaults to {}.

        Returns:
            The response text.

        Raises:
            ValueError: If the conversation ID is invalid.

        Examples:
            >>> assistant = Assistant(api_key="YOUR_API_KEY", model="gpt-3.5-turbo")
            >>> response = assistant.chat("Hello, how are you?")
            >>> print(response)
            Hello, how are you?

        ----------
        """

        convo = self.conversation_id if conv_id is True else str(conv_id)
        if not convo:
            convo = False

        params_for_response = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                input
                                if valid_json == {}
                                else input
                                + "RESPOND ONLY IN VALID JSON FORMAT LIKE THIS: "
                                + json.dumps(valid_json)
                            ),
                        }
                    ],
                }
            ],
            "instructions": self.system_prompt,
            "conversation": convo,
            "max_output_tokens": max_output_tokens,
            "store": store,
            "model": self.model,
            "reasoning": self.reasoning if self.reasoning is not None else None,
            "tools": [],
            "stream": stream,
        }

        if images:
            for i in images:
                params_for_response["input"][0]["content"].append(
                    {
                        "type": "input_image",
                        ("file_id" if i.type == "filepath" else "image_url"): (
                            i.image[2]
                            if not i.type == "Base64"
                            else f"data:image/{i.image[2]}; base64, {i.image[0]}"
                        ),
                    }
                )

        if web_search:
            params_for_response["tools"].append({"type": "web_search"})

        if code_interpreter:
            params_for_response["tools"].append(
                {"type": "code_interpreter", "container": {"type": "auto"}}
            )

        vector: tuple[VectorStore, VectorStore, VectorStores] | None = None
        if file_search:
            vector = self._convert_filepath_to_vector(file_search)

            if file_search_max_searches is None:

                params_for_response["tools"].append(
                    {"type": "file_search", "vector_store_ids": vector[1].id}
                )

            else:
                params_for_response["tools"].append(
                    {
                        "type": "file_search",
                        "vector_store_ids": vector[1].id,
                        "max_searches": file_search_max_searches,
                    }
                )

        params_for_response = {
            k: v for k, v in params_for_response.items() if v is not None
        }

        params_for_response = {
            k: v for k, v in params_for_response.items() if v is not False
        }

        if tools_required == "none":
            params_for_response["tool_choice"] = "none"
        elif tools_required == "auto":
            params_for_response["tool_choice"] = "auto"
        elif tools_required == "required":
            params_for_response["tool_choice"] = "required"

        tool_map: dict[str, types.FunctionType] = {}
        if custom_tools:
            for tool in custom_tools:
                schema = getattr(tool, "schema", None)
                if not schema:
                    raise ValueError(
                        "Custom tools must be decorated with `openai_function`."
                    )

                try:
                    normalized_schema = self._normalize_tool_schema(schema)
                    params_for_response["tools"].append(normalized_schema)
                    tool_name = normalized_schema["function"]["name"]
                    tool_map[tool_name] = tool
                except Exception as e:
                    print("Error adding custom tool: \n", e)
                    print(
                        "\nLine Number : ",
                        e.__traceback__.tb_lineno
                        if isinstance(e, types.TracebackType)
                        else 355,
                    )
                    continue

        params_for_response = {
            k: v for k, v in params_for_response.items() if v is not None
        }

        try:
            if text_stream:
                streaming_params = deepcopy(params_for_response)
                streaming_params["stream"] = True
                return self._text_stream_generator(
                    streaming_params,
                    tool_map,
                    store=store,
                    file_search=bool(file_search),
                    vector=vector,
                )

            request_params = deepcopy(params_for_response)
            request_params.pop("stream", None)

            resp = self.client.responses.create(**request_params)
            resp = self._resolve_function_calls(resp, request_params, tool_map)

        except Exception as e:
            print("Error creating response: \n", e)
            print(
                "\nLine Number : ",
                e.__traceback__.tb_lineno if isinstance(e, types.TracebackType) else 370,
            )  # type: ignore
            return ""

        if store and getattr(resp, "conversation", None):
            self.conversation = resp.conversation

        if file_search and vector:
            vector[2].delete(vector[0].id)

        if return_full_response or stream:
            return resp
        return resp.output_text

    # def function_chat(self, input: str, func: list[Callable], descriptions: type[dict[str, str]]= dict[str, str], temprature: float | None = None):
    #     if temprature is None:
    #         tempratures = self.temperature

    #     that = openai_function(descriptions=descriptions)(func)
    #     if tempratures is None:
    #         tempratures = None
    #         bob = self.asss.create("bob", functions=[that], model=self.model, instructions=self.system_prompt)
    #         blib = bob.conversation.create()
    #         stob = blib.ask(input)

    #     else:
    #         bob = self.asss.create("bob", functions=[that], model=self.model, instructions=self.system_prompt)
    #         blib = bob.conversation.create()
    #         stob = blib.ask(input)

    #     return stob

    def create_conversation(self, return_id_only: bool = False) -> Conversation | str:
        """
        Create a conversation

        Args:
            return_id_only (bool, optional): If True, return only the conversation ID, by default False
        ----------
        """

        conversation = self.client.conversations.create()
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

        clean_params = {
            k: v for k, v in params.items() if v is not None or "" or [] or {}
        }

        try:
            img = self.client.images.generate(**clean_params)

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
        """
        Update the parameters of the assistant.

        Args:
            what_to_change (Literal["model", "system_prompt", "temperature", "reasoning_effort", "summary_length", "function_call_list"]): The parameter to change.
            new_value: The new value for the parameter.

        Returns:
            None

        Raises:
            ValueError: If the parameter to change is invalid.

        Examples:
            >>> assistant.update_assistant("model", "gpt-4o")
            >>> assistant.update_assistant("system_prompt", "You are a helpful assistant.")
            >>> assistant.update_assistant("temperature", 0.7)
            >>> assistant.update_assistant("reasoning_effort", "high")
            >>> assistant.update_assistant("summary_length", "concise")
            >>> assistant.update_assistant("function_call_list", [FunctionCall(name="get_current_weather", arguments={"location": "San Francisco"})])
        """

        if what_to_change == "model":
            self.model = new_value
        elif what_to_change == "system_prompt":
            self.system_prompt = new_value
        elif what_to_change == "temperature":
            self.temperature = new_value
        elif what_to_change == "reasoning_effort":
            self.reasoning_effort = new_value
        elif what_to_change == "summary_length":
            self.summary_length = new_value
        elif what_to_change == "function_call_list":
            self.function_call_list = new_value
        else:
            raise ValueError("Invalid parameter to  change")

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
        """
        Convert text to speech

        Args:
            input (str): The text to convert to speech
            model (Literal['tts-1', 'tts-1-hd', 'gpt-4o-mini-tts'], optional): The model to use. Defaults to "tts-1".
            voice (str | Literal['alloy', 'ash', 'ballad', 'coral', 'echo', 'sage', 'shimmer', 'verse', 'marin', 'cedar'], optional): The voice to use. Defaults to "alloy".
            instructions (str, optional): The instructions to follow. Defaults to "NOT_GIVEN".
            response_format (Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'], optional): The response format to use. Defaults to "wav".
            speed (float, optional): The speed to use. Defaults to 1.
            play (bool, optional): Whether to play the audio. Defaults to True.
            save_to_file_path (str | None, optional): The path to save the audio to. Defaults to None.


        Returns:
            None

        Raises:
            None

        Examples:
            ```python
                assistant.text_to_speech(input="hello", voice="alloy", save_to_file_path="test.wav", response_format="wav")
            ```

            ```python
                assistant.text_to_speech(input="hello", voice="alloy", response_format="wav", play=True)
            ```

            ```python
                assistant.text_to_speech(input="hello", voice="alloy", response_format="wav", play=True, save_to_file_path="test.wav")
            ```
        """
        params = {
            "input": input,
            "model": model,
            "voice": voice,
            "instructions": instructions,
            "response_format": response_format,
            "speed": speed,
        }

        respo = self.client.audio.speech.create(**params)

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
        file_search: list[str] | None = None,
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
        """
        This is the full text to speech function.
        Args:
            input: The input text. Defaults to True.
            conv_id: The conversation ID. defaults to True. If True, a the default conversation ID will be used.
            max_output_tokens: The maximum output tokens. Defaults to None.
            store: Whether to store the conversation. Defaults to False.
            web_search: Whether to use web search. Defaults to None.
            code_interpreter: Whether to use code interpreter. Defaults to None.
            file_search: The file search. Defaults to None.
            tools_required: The tools required. Defaults to "auto".
            model: The model. Defaults to "tts-1".
            voice: The voice.   Defaults to "alloy".
            instructions: The instructions. Defaults to "NOT_GIVEN".
            response_format: The response format. Defaults to "wav".
            speed: The speed. Defaults to 1.
            play: Whether to play the audio. Defaults to True.
            print_response: Whether to print the response. Defaults to False.
            save_to_file_path: The save to file path. Defaults to None.


        Returns:
            The response.

        Raises:
            Exception: If the response format is not wav.

        Example:
            ```python
            >>> assistant.full_text_to_speech("Hello, world!", model="tts-1", voice="alloy", instructions="NOT_GIVEN", response_format="wav", speed=1, play=True, save_to_file_path=None)
            ```

            ```python
            >>> assistant.full_text_to_speech("Hello, world!", model="tts-1", voice="alloy", instructions="NOT_GIVEN", response_format="wav", speed=1, play=True, save_to_file_path="test.wav")
            ```
        """
        param = {
            "input": input,
            "conv_id": conv_id,
            "max_output_tokens": max_output_tokens,
            "store": store,
            "web_search": web_search,
            "code_interpreter": code_interpreter,
            "file_search": file_search,
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
        stt_model = stt.STT(
            model=model, aggressive=aggressive, chunk_duration_ms=chunk_duration_ms
        )

        if mode == "keyboard":
            result = stt_model.record_with_keyboard(log=log_directions, key=key)
        elif mode == "vad":
            result = stt_model.record_with_vad(log=log_directions)

        elif isinstance(mode, Seconds):
            result = stt_model.record_for_seconds(mode)

        return result

    class __mass_update_helper(TypedDict, total=False):
        model: ResponsesModel
        system_prompt: str
        temperature: float
        reasoning_effort: Literal["minimal", "low", "medium", "high"]
        summary_length: Literal["auto", "concise", "detailed"]
        function_call_list: list[types.FunctionType]

    def mass_update(self, **__mass_update_helper: Unpack[__mass_update_helper]):
        for key, value in __mass_update_helper.items():
            setattr(self, key, value)


if __name__ == "__main__":
    bob: Assistant = Assistant(
        api_key=None, model="gpt-4o", system_prompt="You are a helpful assistant."
    )

    print(
        bob.speech_to_text(mode="vad", model="gpt-4o-transcribe", log_directions=True)
    )
