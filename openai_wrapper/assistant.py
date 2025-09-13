import os
import types
from typing import Any, Literal, TypeAlias, Unpack
from openai import OpenAI, pydantic_function_tool
from openai.types.shared_params import ResponsesModel, Reasoning
from os import getenv
from typing_extensions import TypedDict
from openai.types.conversations.conversation import Conversation
from openai.types.vector_store import VectorStore
from openai.resources.vector_stores.vector_stores import VectorStores
import base64
from pydantic import BaseModel, ValidationError, create_model

# e.g. {"type": "string"}
PropertySpec: TypeAlias = dict[str, str]
# e.g. {"foo": {"type": "string"}}
Properties: TypeAlias = dict[str, PropertySpec]

Parameters: TypeAlias = dict[str, str | Properties | list[str]]
FunctionSpec: TypeAlias = dict[str, str | Parameters]
ToolSpec: TypeAlias = dict[str, str | FunctionSpec]


class Assistant:
    def __init__(
        self,
        api_key: str | None,
        model: ResponsesModel,
        system_prompt: str = "",
        default_conversation: Conversation | bool = True,
        temperature: float | None = None,
        reasoning_effort: Literal["minimal",
                                  "low", "medium", "high"] = "medium",
        summary_length: Literal["auto", "concise", "detailed"] = "auto",
    ):
        self.model = model
        if not api_key:
            if not getenv("OPENAI_API_KEY"):
                raise ValueError("No API key provided.")
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

        if default_conversation:
            self.conversation = self.create_conversation()
            self.conversation_id = self.conversation.id  # type: ignore
        else:
            self.conversation = None
            self.conversation_id = None

    def chat(
        self,
        input: str,
        custom_tools: list[Any] | None = None,
        valid_json: dict | None = None,
        force_valid_json: bool = False,
    ):
        params = {
            "model": self.model,
            "input": input,
            "instructions": self.system_prompt,
            "tools": [],
        }

        tools: list = []
        if custom_tools:
            for fn in custom_tools:
                # Instead of giving a plain function, wrap it into a BaseModel schema first
                class FnModel(BaseModel):
                    prompt: str

                tools.append(pydantic_function_tool(FnModel))

        if tools:
            params["tools"] = tools

        clean_params = {k: v for k, v in params.items() if v}

        if valid_json and force_valid_json:
            def make_model_from_dict(name: str, data: dict[str, Any]) -> type[BaseModel]:
                fields: dict[str, tuple[type, Any]] = {}
                for k, v in data.items():
                    fields[k] = (type(v), ...)
                return create_model(name, **fields)  # type: ignore

            JSONModel = make_model_from_dict("JSONModel", valid_json)
            response = self.client.responses.parse(
                **clean_params, text_format=JSONModel)
        else:
            response = self.client.responses.create(**clean_params)

        return response.output_text

    def create_conversation(self, return_id_only: bool = False) -> Conversation | str:
        conversation = self.client.conversations.create()
        if return_id_only:
            return conversation.id
        return conversation


if __name__ == "__main__":
    bob = Assistant(api_key=None, model="gpt-4o",
                    system_prompt="You are a helpful assistant.")

    def get_goofy_prompt(prompt: str) -> str:
        return prompt

    print(
        bob.chat(
            "get the goofy prompt use tools",
            valid_json={"answer": "str"},
            force_valid_json=True,
            custom_tools=[get_goofy_prompt],
        )
    )
