# Easy‑gpt

Lightweight helpers around OpenAI's Python SDK to make chatting with the Responses API, maintaining Conversations, uploading files to Vector Stores for file-search, and generating images easier via a single `Assistant` class.

Note: the package folder name contains a hyphen. Use `importlib` to import it (see Quickstart) or import the file relatively inside your own package.

## Features

- Assistant: simple chat with Conversations and tool options
- File search: upload local files to a temporary Vector Store
- Code interpreter and web search toggles (when supported)
- Image generation with `gpt-image-1`, `dall-e-2`, and `dall-e-3`
- Small utility to define custom function-calling tool specs

## Installation

- Requires Python 3.10+
- Install the OpenAI client and this package in editable mode:

  - `pip install -U openai typing_extensions`
  - `pip install -e .`

Set your API key as `OPENAI_API_KEY` in the environment (or pass `api_key` to `Assistant`).

## Quickstart

```python
from importlib import import_module

# Import because the package directory name has a hyphen
assistant_mod = import_module('Easy-gpt.assistant')
Assistant = assistant_mod.Assistant

bot = Assistant(api_key=None, model="gpt-4o", system_prompt="You are helpful.")
text, conversation = bot.chat("Summarize: The sky is blue due to Rayleigh scattering.")
print(text)
```

## Chat options (summary)

`Assistant.chat(input, conv_id=True, max_output_tokens=None, store=False, web_search=None, code_interpreter=None, file_search=None, tools_required="auto", if_file_search_max_searches=50, return_full_response=False, valid_json=None)`

- input: user text
- conv_id: `True` uses the default conversation; pass a `Conversation` or id; or `None`/`False` to not use one
- file_search: list of file paths to index temporarily
- web_search / code_interpreter: enable tools where supported
- tools_required: "none" | "auto" | "required"
- return_full_response: return full SDK object alongside `output_text`
- valid_json: JSON schema hint appended to the prompt

Returns either `[text, conversation]` or `[response, text]` when `return_full_response=True`.

## Image generation

```python
img_b64 = bot.image_generation(
    prompt="A watercolor fox in a misty forest",
    model="gpt-image-1",
    output_format="png",
    size="1024x1024",
    return_base64=True,
)
```

## License

MIT
