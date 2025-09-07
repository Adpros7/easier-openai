import os
import types
from typing import Any, Callable, Literal, Sequence, TypeAlias, Unpack
import dotenv
from openai import OpenAI
from openai.types.shared_params import ResponsesModel, Reasoning
from os import getenv
from typing_extensions import TypedDict
import random

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
    def __init__(self, api_key: str | None, model: ResponsesModel, system_prompt: str = "", temperature: float | None = None, reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium", summary_length: Literal["auto", "concise", "detailed"] = "auto"):
        """Initialize the Assistant with configuration parameters. ONLY USE REASONING  WITH GPT-5 and o MODELS."""
        self.model = model
        if not api_key:
            if not getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "No API key provided. Please set the OPENAI_API_KEY environment variable or provide an api_key argument.")
            else:
                self.api_key = getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key
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


        self.client = OpenAI(api_key=self.api_key)

    def chat(
        self,
        input: str = "",
        conversation: str | None = None,           # was `id`; Responses expects `conversation`
        enable_web_search: bool | None = None,     # ignored; not an official tool in Responses
        file_search: bool | None = None,           # enable the tool (boolean)
        code_interpreter: bool | None = None,      # enable the tool (boolean)
        image_generation: bool | None = None,      # ignored; not a Responses tool
        tools: list[Tool] | None = None,           # must be a list
        store: bool | None = None,
        max_output_tokens: int | None = None,
        upload_files: Iterable[tuple[str, bytes]] | None = None,  # (filename, data)
        return_full_response: bool | None = None
    ) -> Any:
        """
        Create a Responses API call with optional tools and file uploads.
        - `conversation`: continue a stored conversation id (use with store=True).
        - `file_search` / `code_interpreter`: toggles for official tools.
        - `upload_files`: iterable of (filename, bytes) to attach as input files.
        - `return_full_response`: if True, return SDK object. Else [text, id].
        """
        # Assemble tool list
        merged_tools: list[Tool] = []
        if tools:
            if not isinstance(tools, list):
                raise TypeError("`tools` must be a list[dict].")
            merged_tools.extend(tools)

        if code_interpreter:
            merged_tools.append({"type": "code_interpreter"})
        if file_search:
            merged_tools.append({"type": "file_search"})

        # Build input parts
        input_parts: list[dict[str, Any]] = [{"type": "input_text", "text": input}]

        # Optional file uploads become input parts
        if upload_files:
            for fname, data in upload_files:
                f = self.client.files.create(file=(fname, data), purpose="assistants")
                input_parts.append({"type": "input_file", "file_id": f.id})

        # Build API kwargs, excluding local-only flags
        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_parts,
        }
        if conversation is not None:
            api_kwargs["conversation"] = conversation
        if merged_tools:
            api_kwargs["tools"] = merged_tools
        if store is not None:
            api_kwargs["store"] = store
        if max_output_tokens is not None:
            api_kwargs["max_output_tokens"] = max_output_tokens

        # Fire the call
        resp = self.client.responses.create(**api_kwargs)

        # Return mode
        if return_full_response:
            return resp
        return [resp.output_text, resp.id]

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
    bob = Assistant(api_key=None, model="gpt-5-nano",
                    system_prompt="You are a helpful assistant.")
    while True:
        user_input = input("User: ")
        response = bob.chat(input=user_input, store_id=True)
        print("Assistant:", response[0])
