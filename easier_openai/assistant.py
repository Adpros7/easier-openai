from typing import Optional
from openai import OpenAI
from .models import Models

class Assistant:
    def __init__(self, api_key: Optional[str] = None, models: Models = "gpt-5.5"):
        self.client = OpenAI()