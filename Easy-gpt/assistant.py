import os
import types
from typing import Any, Callable, Literal, Sequence, Unpack
import dotenv
from openai import OpenAI
from openai.types.shared_params import ResponsesModel, Reasoning
from os import getenv
from typing_extensions import TypedDict
import random


class Assistant:
    def __init__(self, api_key: str | None, model: ResponsesModel, system_prompt: str = "", temperature: float | None = None, reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium", summary_length: Literal["auto", "concise", "detailed"] = "auto", function_call_list_with_descriptions: None | dict[Callable, str] = None):
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
        if reasoning_effort and summary_length:
            self.reasoning = Reasoning(
                effort=reasoning_effort, summary=summary_length)
        if temperature is None:
            self.temperature = None
            
        else:
            self.temperature = temperature
        self.function_call_list = function_call_list_with_descriptions or {}

        self.client = OpenAI(api_key=self.api_key)

    def chat(self, input: str = "", id: str | None = None, web_search: bool | None = None,  store_id: bool | None = None, max_output_tokens: int | None = None, return_full_response: bool | None = None) -> Any:
        params = {"id": id,
                  "tools": tools,
                  "store": store_id,
                  "max_output_tokens": max_output_tokens,
                  "return_full_response": return_full_response}
        clean_params = {k: v for k, v in params.items() if v is not None}
        
        response = self.client.responses.create(
            model=self.model,
            input=input,
            **clean_params
        )
        
        if return_full_response:
            return response
        else:
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
    bob = Assistant(api_key=dotenv.get_key(r"C:\Users\prani\OneDrive\Desktop\Coding\AI\ChatPPT\ChattGpt-with live video\API_KEY.env", "OPENAI_API_KEY"), model="gpt-5-nano", system_prompt="You are a helpful assistant." )
    while True:
        user_input = input("User: ")
        response = bob.chat(input=user_input, store_id=True)
        print("Assistant:", response[0])