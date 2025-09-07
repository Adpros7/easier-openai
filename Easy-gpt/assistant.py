import os
import types
from typing import Any, Callable, Literal, Sequence, Unpack
from openai import OpenAI
from openai.types.shared_params import ResponsesModel, Reasoning
from os import getenv
from typing_extensions import TypedDict
import random


class Assistant:
    def __init__(self, api_key: str, model: ResponsesModel, system_prompt: str = "", temperature: float = 0.7, reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium", summary_length: Literal["auto", "concise", "detailed"] = "auto", function_call_list: None | list[types.FunctionType] = None):
        """Initialize the Assistant with configuration parameters. ONLY USE REASONING  WITH GPT-5 and o MODELS."""
        self.model = model
        if not api_key:
            raise ValueError("API key is required")
        if api_key.startswith("sk-"):
            self.api_key = api_key
        elif os.getenv(api_key):
            self.api_key = os.getenv(api_key)
        else:
            raise ValueError("Invalid API key or environment variable name")

        self.system_prompt = system_prompt
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.summary_length = summary_length
        self.reasoning = Reasoning(
            effort=reasoning_effort, summary=summary_length)
        self.function_call_list = function_call_list or []

        self.client = OpenAI(api_key=self.api_key)

    def chat(self, input: str = "", id: str | None = None, tools: Literal["web search", "code interpreter", "file search", "image generation"] | None = None, store_id: bool = False, return_full_response: bool = False) -> Any:

        if not tools and not self.function_call_list:
            if id:
                response = self.client.responses.create(
                    model=self.model,
                    input=input,
                    instructions=self.system_prompt,
                    temperature=self.temperature,
                    reasoning=self.reasoning,
                    store=store_id,
                    conversation=id
                )
                
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=input,
                    instructions=self.system_prompt,
                    temperature=self.temperature,
                    reasoning=self.reasoning,
                    store=store_id
                )
                
        
            if store_id and return_full_response:
                return [response.output_text, response.id, response]
            elif store_id and not return_full_response:
                return [response.output_text, response.id]
            elif not store_id and return_full_response:
                return [response.output_text, response]
            else:
                return response.output_text

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



