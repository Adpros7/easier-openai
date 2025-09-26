# Easier OpenAI

Easier OpenAI offers a light wrapper around the OpenAI Python SDK so you can chat with models, toggle built-in tools, upload reference files, and request images through one unified helper.

## Highlights

- Single `Assistant` class for conversational flows, tool selection, and multi-turn context.
- Temporary vector store support to search across local reference documents during a chat.
- Flags for web search or code interpreter when the selected OpenAI model supports them.
- Image helpers for `gpt-image-1`, `dall-e-2`, and `dall-e-3` with convenient file handling.

## Installation

```bash
pip install easier-OpenAI
```

Set `OPENAI_API_KEY` in your environment or pass an explicit key when you build the assistant instance.

## Usage Example

```python
from openai import Assistant

assistant = Assistant(model="gpt-4o", system_prompt="You are concise.")
response_text = assistant.chat("Summarize Rayleigh scattering in one sentence.")
print(response_text)
```

### File Search Example

```python
notes = ["notes/overview.md", "notes/data-sheet.pdf"]
reply = assistant.chat(
    "Highlight key risks from the attached docs",
    file_search=notes,
    tools_required="auto",
)
print(reply)
```

## Requirements

- Python 3.10 or newer
- `openai>=1.43.0`
- `typing_extensions>=4.7.0`
- `pydantic>=2.0.0`

## Contributing

Issues and pull requests are welcome. Please run checks locally before submitting changes.

## License

This project is licensed under the [Apache License 2.0](LICENSE).