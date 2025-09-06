from typing import Literal
from openai import OpenAI


class assistant:
    def __init__(self, api_key: str | None, system_prompt: str, model: Literal['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07', 'gpt-5-chat-latest', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4.1-2025-04-14', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-nano-2025-04-14', 'o4-mini', 'o4-mini-2025-04-16', 'o3', 'o3-2025-04-16', 'o3-mini', 'o3-mini-2025-01-31', 'o1', 'o1-2024-12-17', 'o1-preview', 'o1-preview-2024-09-12', 'o1-mini', 'o1-mini-2024-09-12', 'gpt-4o', 'gpt-4o-2024-11-20', 'gpt-4o-2024-08-06', 'gpt-4o-2024-05-13', 'gpt-4o-audio-preview', 'gpt-4o-audio-preview-2024-10-01', 'gpt-4o-audio-preview-2024-12-17', 'gpt-4o-audio-preview-2025-06-03', 'gpt-4o-mini-audio-preview', 'gpt-4o-mini-audio-preview-2024-12-17', 'gpt-4o-search-preview', 'gpt-4o-mini-search-preview', 'gpt-4o-search-preview-2025-03-11', 'gpt-4o-mini-search-preview-2025-03-11', 'chatgpt-4o-latest', 'codex-mini-latest', 'gpt-4o-mini', 'gpt-4o-mini-2024-07-18', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4-0125-preview', 'gpt-4-turbo-preview', 'gpt-4-1106-preview', 'gpt-4-vision-preview', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-4-32k-0613', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-16k-0613', 'o1-pro', 'o1-pro-2025-03-19', 'o3-pro', 'o3-pro-2025-06-10', 'o3-deep-research', 'o3-deep-research-2025-06-26', 'o4-mini-deep-research', 'o4-mini-deep-research-2025-06-26', 'computer-use-preview', 'computer-use-preview-2025-03-11'],
                 context: bool = False
                 ):
        if not api_key:
            self.client = OpenAI()
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.sys_prompt = system_prompt
        self.context = {"user": [], "assistant": []}
        if context:
            self.sys_prompt += f"Respond in only valid python list like this: ['actual response', [list of user context points formatted as a python list], [assistant's context points formatted as a python list]]. Here is the context you can use: {context}"
        
    def ask(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            instructions=self.sys_prompt
        )
        if self.context:
            self.context["user"].append(response.output_text[1])
            self.context["assistant"].append(response.output_text[2])
            return str(response.output_text[0])
        
        else:
            return str(response.output_text)
    
    def ask_stream(self, prompt: str, var_to_update: str):
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            instructions=self.sys_prompt,
            stream=True
        )
        bob = eval(var_to_update)
        for chunk in response:
           if hasattr(chunk, 'output_text'):
                bob += str(chunk.output_text)
                yield bob
        
if __name__ == "__main__":
    var = ""
    assistants = assistant(api_key=None, system_prompt="You are a helpful assistant.", model="gpt-4o-mini", context=False)
    for i in assistants.ask_stream("Write a poem about a computer", "var"):
        print(i)
