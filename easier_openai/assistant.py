from httpx import URL
from openai.types.conversations.conversation import Conversation
from typing import Optional
from openai import OpenAI
from .models import Model

class Assistant:
    def __init__(self, api_key: Optional[str] = None, model: Model = "gpt-5.5", base_url: str = "https://api.openai.com/v1", conversation: bool = True):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.conversation: Conversation | None = self.client.conversations.create() if conversation else None
        self.model = model
    
    def change_model(self, model: Model):
        self.model = model
    
    def new_conversation(self):
        self.conversation = self.client.conversations.create()
    
    def change_base_url(self, new_url: str):
        self.client._base_url = URL(new_url)
    
    def 