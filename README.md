# Easy GPT

Easy GPT is a tiny wrapper around the OpenAI Responses API that aims to make
chatting with OpenAI models simple and flexible.

## Features

- Choose any OpenAI model by name.
- Optional streaming in both synchronous and asynchronous contexts.
- Support for `AsyncOpenAI` via async helper methods.
- Multiple assistants with independent system prompts.
- Update system prompts at runtime with `update_system_prompt`.
- Register custom function tools through `add_tool`.
- Context tracking and simple garbage collection using `clear_context`.
- API key pulled from the environment when not supplied explicitly.

- Typed model hints via `ModelName` literals and faster startup thanks to
  shared OpenAI clients and `dataclass` slots.
=======

## Quick start

```python

from easy_gpt import Assistant, ModelName

# IDEs will offer suggestions for ModelName values
assistant = Assistant(system_prompt="You are helpful.", model="gpt-4o-mini")
=======
from easy_gpt import Assistant

assistant = Assistant(system_prompt="You are helpful.")

print(assistant.ask("Hello, world!"))

for partial in assistant.ask_stream("Write a limerick about Python"):
    print(partial)
```

## Async usage

```python
import asyncio
from easy_gpt import Assistant

async def main():
    bot = Assistant(system_prompt="You are helpful.")
    reply = await bot.ask_async("What is 2 + 2?")
    print(reply)

asyncio.run(main())
```

## License

MIT
=======


