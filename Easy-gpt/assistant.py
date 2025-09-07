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
    def __init__(self, api_key: str | None, model: ResponsesModel, system_prompt: str = "", temperature: float | None = None, reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium", summary_length: Literal["auto", "concise", "detailed"] = "auto", function_call_list: None | ToolSpec = None):
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
        self.function_call_list = function_call_list or []

        self.client = OpenAI(api_key=self.api_key)

    def chat(
        self,
        input: str = "",
        id: str | None = None,
        web_search: bool | None = None,
        file_search: str | bytes | None = None,
        code_interpreter: bool | None = None,
        image_generation: bool | None = None,
        tools: list[dict] | None = None,   # expects structured tools
        store_id: bool | None = None,
        max_output_tokens: int | None = None,
        return_full_response: bool | None = None
    ) -> Any:

        # build params
        params = {
            "id": id,
            "tools": tools,
            "store": store_id,
            "max_output_tokens": max_output_tokens,
            "return_full_response": return_full_response,
        }
        clean_params = {k: v for k, v in params.items() if v is not None}

        # handle built-in toggles as tool objects
        built_in_tools = []
        if web_search:
            built_in_tools.append({"type": "web_search"})
        if code_interpreter:
            built_in_tools.append({"type": "code_interpreter"})
        if file_search:
            built_in_tools.append({"type": "file_search"})
        if image_generation:
            built_in_tools.append({"type": "image_generation"})

        # merge built-ins with custom tools if any
        if built_in_tools or tools:
            clean_params["tools"] = (tools or []) + built_in_tools

        # create response
        response = self.client.responses.create(
            model=self.model,
            input=input,
            **clean_params
        )

        # return mode
        if return_full_response:
            return response
        return [response.output_text, response.id]

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
