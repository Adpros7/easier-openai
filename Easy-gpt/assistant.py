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
            self.conversation_id = self.conversation.id # type: ignore
        else:
            self.conversation = None
            self.conversation_id = None


    def _convert_filepath_to_vector(self, filepath: str) -> VectorStore:
        # Step 1: create the vector store
        filename = os.path.basename(filepath)
        vector_store = self.client.vector_stores.create(name=filename)
        

        # Step 2: upload the file into the vector store
        with open(filepath, "rb") as f:
            self.client.vector_stores.files.upload_and_poll(
                vector_store_id=vector_store.id,
                file=f
            )

        return vector_store

    

    def chat(
        self,
        input: str,
        conv_id: str | None | Conversation | bool = True,
        max_output_tokens: int | None = None,
        store: bool | None = False,
        web_search: bool | None = None,
        code_interpreter: bool | None = None,
        image_generator: bool | None = None,
        file_search: list[str | bytes] | None = None,
        return_full_response: bool = False,

    ):
        """Reasoning can only have gpt 5 and o and temp only to the big boy models set conv_id to True to use default conversation"""
        params = {
            "model": self.model,
            "input": input,
            "instructions": self.system_prompt if self.system_prompt else "",
            "temperature": self.temperature if self.temperature else None,
            "max_output_tokens": max_output_tokens if max_output_tokens else None,
            "store": store if store else None,
            "conversation": conv_id if conv_id is str else None,
            "return_full_response": return_full_response,
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
        
        clean_params = {k: v for k, v in params.items() if v is not None or "" or [] or {}}
        clean_params.__delitem__("return_full_response")
        response = self.client.responses.create(
            **clean_params

        )

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

    while True:
        user_input = input("User: ")
        # Only send id if it's valid
        response = bob.chat(user_input, conv_id=True, web_search=True)

        print("Assistant:", response[0])
        print("Conv ID:", response[1])
