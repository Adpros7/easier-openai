from openai.types.responses.response import Response
from httpx import URL
from openai.types.conversations.conversation import Conversation
from typing import Optional
from openai import OpenAI
from .models import Model

class Assistant:
    def __init__(self, instructions: str = "", api_key: Optional[str] = None, model: Model = "gpt-5.5", base_url: str = "https://api.openai.com/v1", conversation: bool = True):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.conversation: Conversation | None = self.client.conversations.create() if conversation else None
        self.model = model
        self.instructions: str = instructions
    
    def change_model(self, model: Model):
        self.model = model
    
    def new_conversation(self):
        self.conversation = self.client.conversations.create()
    
    def change_base_url(self, new_url: str):
        self.client._base_url = URL(new_url)
    
    def change_instructions(self, new_instructions: str):
        self.instructions: str = new_instructions
    
    def chat(self, input, long_running: bool = False, return_full_response: bool = False, stream: bool = False):  
        if not long_running:
            out: Response = self.client.responses.create(
                conversation=self.conversation.id if self.conversation else None,
                input=input,
                instructions=self.instructions,
            )

            return out if return_full_response else out.output_text

        else:
            if stream:
                raise NotImplementedError("Streaming support with long running taks (background mode) is not supported. Its not on the todo list either, but feel free to open a PR")
            
            