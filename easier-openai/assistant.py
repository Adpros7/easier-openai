from openai.types import ResponsesModel
from typing import Optional, TypeAlias
from openai import OpenAI


Models: TypeAlias = ResponsesModel

class Assistant:
    def __init__(self, api_key: Optional[str] = None):
        pass


print(OpenAI().realtime.connect())
