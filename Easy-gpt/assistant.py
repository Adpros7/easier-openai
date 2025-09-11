import os
import types
from typing import Any, Callable, Literal, Sequence, TypeAlias, Unpack
import dotenv
from openai import OpenAI
from openai.types.shared_params import ResponsesModel, Reasoning
from os import getenv
from typing_extensions import TypedDict
from openai.types.responses.response_conversation_param import ResponseConversationParam
from openai.types.conversations.conversation import Conversation
from openai.types.vector_store import VectorStore
from openai.resources.vector_stores.vector_stores import VectorStores
import base64

# e.g. {"type": "string"}
PropertySpec: TypeAlias = dict[str, str]
# e.g. {"foo": {"type": "string"}}
Properties: TypeAlias = dict[str, PropertySpec]

Parameters: TypeAlias = dict[str, str | Properties | list[str]]
FunctionSpec: TypeAlias = dict[str, str | Parameters]
ToolSpec: TypeAlias = dict[str, str | FunctionSpec]


class CustomToolInputFormat:
    def __init__(self, fn_name: str, description: str, parameters: list[str], required_params: list[str]):
        self.tool = {
            "type": "function",
            "function": {
                "name": fn_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {param: {"type": "string"} for param in parameters},
                    "required": required_params
                }
            }
        }

    def to_dict(self) -> ToolSpec:
        return self.tool


class Assistant:
    def __init__(self, api_key: str | None, model: ResponsesModel, system_prompt: str = "", default_conversation: Conversation | bool = True, temperature: float | None = None, reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium", summary_length: Literal["auto", "concise", "detailed"] = "auto"):
        """Initialize the Assistant with configuration parameters. ONLY USE REASONING  WITH GPT-5 and o MODELS. Temp only with bigger models"""
        self.model = model
        if not api_key:
            if not getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "No API key provided. Please set the OPENAI_API_KEY environment variable or provide an api_key argument.")
            else:
                self.api_key = getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.summary_length = summary_length
        if reasoning_effort and summary_length:
            self.reasoning = Reasoning(
                effort=reasoning_effort, summary=summary_length)
        if temperature is None:
            self.temperature = None

        else:
            self.temperature = temperature

        if default_conversation:
            self.conversation = self.create_conversation()
            self.conversation_id = self.conversation.id  # type: ignore
        else:
            self.conversation = None
            self.conversation_id = None

    def _convert_filepath_to_vector(self, list_of_files: list[str]) -> tuple[VectorStore, VectorStore, VectorStores]:
        # Create a single vector store and upload all files into it.
        if not isinstance(list_of_files, list) or len(list_of_files) == 0:
            raise ValueError(
                "list_of_files must be a non-empty list of file paths.")
        for filepath in list_of_files:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
        vector_store_create = self.client.vector_stores.create(
            name="vector_store")
        vector_store = self.client.vector_stores.retrieve(
            vector_store_create.id)
        vector = self.client.vector_stores
        for filepath in list_of_files:
            with open(filepath, "rb") as f:
                self.client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store_create.id,
                    file=f
                )

        return vector_store_create, vector_store, vector

    def chat(
        self,
        input: str,
        conv_id: str | None | Conversation | bool = True,
        max_output_tokens: int | None = None,
        store: bool | None = False,
        web_search: bool | None = None,
        code_interpreter: bool | None = None,
        file_search: list[str] | None = None,
        if_file_search_max_searches: int | None = 50,
        return_full_response: bool = False,

    ):
        """Reasoning can only have gpt 5 and o and temp only to the big boy models set conv_id to True to use default conversation For image gen True is forced tool call, False is not forced tool call, and None is no tool call"""
        params = {
            "model": self.model,
            "input": input,
            "instructions": self.system_prompt if self.system_prompt else "",
            "temperature": self.temperature if self.temperature else None,
            "max_output_tokens": max_output_tokens if max_output_tokens else None,
            "store": store if store else None,
            "conversation": conv_id if isinstance(conv_id, str) else None,
            "tools": []
        }

        if params["conversation"] is None:
            if conv_id is True:
                params["conversation"] = self.conversation_id
            elif isinstance(conv_id, Conversation):
                params["conversation"] = conv_id.id
            else:
                params["conversation"] = None

        if web_search:
            params["tools"].append({"type": "web_search"})

        if file_search:
            vstore = self._convert_filepath_to_vector(file_search)
            params["tools"].append({
                "type": "file_search",
                "vector_store_ids": [vstore[0].id],
                "max_num_results": if_file_search_max_searches if if_file_search_max_searches else 50,
                "tool_choice": {}
            })

        clean_params = {k: v for k, v in params.items(
        ) if v is not None or "" or [] or {}}
        response = self.client.responses.create(
            **clean_params

        )
        if file_search:
            vstore[2].delete(vstore[1].id)  # Free up memory

        if return_full_response:
            return [response, response.output_text]

        else:
            return [response.output_text, response.conversation]

    def create_conversation(self, return_id_only: bool = False) -> Conversation | str:
        """Create a new conversation on the OpenAI server."""
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
        quality: Literal['standard', 'hd', 'low',
                         'medium', 'high', 'auto'] | None = None,
        size: Literal['auto', '1024x1024', '1536x1024', '1024x1536',
                      '256x256', '512x512', '1792x1024', '1024x1792'] | None = None,
        n: int = 1,
        moderation: Literal["auto", "low"] | None = None,
        style: Literal["vivid", "natural"] | None = None,
        return_base64: bool = False,
        make_file: bool  = False,
        file_name_if_make_file: str = "generated_image",

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

        clean_params = {k: v for k, v in params.items(
        ) if v is not None or "" or [] or {}}
        
        try:
            img = self.client.images.generate(
                **clean_params

            )
            
        except Exception as e:
            raise e
        
        
        if return_base64 and not make_file:
            return img.data[0].b64_json
        elif make_file and not return_base64:
            image_data = img.data[0].b64_json
            with open(file_name_if_make_file, "wb") as f:
                f.write(base64.b64decode(image_data))
        else: 
            image_data = img.data[0].b64_json
            name = file_name_if_make_file + "." + output_format
            with open(name, "wb") as f:
                f.write(base64.b64decode(image_data))
                
            return img.data[0].b64_json

    def update_assistant(self, what_to_change: Literal["model", "system_prompt", "temperature", "reasoning_effort", "summary_length", "function_call_list"], new_value):
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
    bob = Assistant(api_key=None, model="gpt-4o",
                    system_prompt="You are a helpful assistant.")

    # Create a conversation on the OpenAI server

    print(bob.image_generation(prompt="bob the bannana", model="dall-e-3", return_base64=True, make_file=True, file_name_if_make_file="bob_the bannana"))
