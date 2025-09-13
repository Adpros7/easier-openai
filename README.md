# Easy‑GPT

Lightweight helpers around OpenAI’s Python SDK to chat with the Responses API, maintain Conversations, use Vector Stores for file‑search, and generate images — all via a single `Assistant` class.

## Features

- Assistant: simple chat with Conversations and tool options
- File search: upload local files into a temporary Vector Store
- Code interpreter and web search toggles (when supported)
- Image generation with `gpt-image-1`, `dall-e-2`, and `dall-e-3`
- Utility to define custom function‑calling tool specs

## Installation

- Python 3.10+
- `pip install easy-gpt`

Set your API key as `OPENAI_API_KEY` in the environment (or pass `api_key` to `Assistant`).

## Quickstart

```python
from openai_wrapper import Assistant

bot = Assistant(api_key=None, model="gpt-4o", system_prompt="You are helpful.")
reply = bot.chat("Summarize: The sky is blue due to Rayleigh scattering.")
print(reply)
```

## Chat options (summary)

`Assistant.chat(input, conv_id=True, max_output_tokens=None, store=False, web_search=None, code_interpreter=None, file_search=None, custom_functions=None, tools_required="auto", if_file_search_max_searches=50, return_full_response=False, valid_json=None, force_valid_json=False)`

- input: user text
- conv_id: `True` uses a default conversation; pass a `Conversation`/id; or `None`/`False` for no conversation
- file_search: list of local file paths to index temporarily
- web_search / code_interpreter: enable tools where supported
- custom_functions: Python callables or prebuilt `CustomToolParam` specs
- tools_required: "none" | "auto" | "required"
- valid_json / force_valid_json: hint or enforce structured JSON output
- return_full_response: return full SDK object alongside `output_text`

Returns either just the output text, or `[response, text]` when `return_full_response=True`.

## Image generation

```python
img_b64 = bot.image_generation(
    prompt="A watercolor fox in a misty forest",
    model="gpt-image-1",
    output_format="png",
    size="1024x1024",
    return_base64=True,
)

# Optionally write to a file if you decode base64 yourself
# or use make_file=True and file_name_if_make_file="my_image".
```

## Notes

- Environment variable: `OPENAI_API_KEY` must be set unless you pass `api_key` directly.
- Requires the OpenAI Python SDK `openai>=1.43.0` (installed automatically).

## License

MIT

