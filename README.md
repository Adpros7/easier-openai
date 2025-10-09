# Easier OpenAI

Easier OpenAI wraps the official OpenAI Python SDK so you can drive modern assistants, manage tool selection, search files, and work with speech from one helper package -- all with minimal boilerplate.

## Features
- High level `Assistant` wrapper with conversation memory, tool toggles, and streaming helpers.
- Temporary vector store ingestion to ground answers in local notes or documents.
- Text to speech and speech to text bridges designed for quick experiments or internal tooling.
- Lazy module loading so `import easier_openai` stays fast even as optional helpers expand.
- Optional `openai_function` decorator and schema helpers to make tool definition repeatable.
- Type hints and comprehensive inline docstrings across the project for easier discovery.

## Installation
```bash
pip install easier-openai
```

Optional extras:
```bash
pip install "easier-openai[function_tools]"   # decorator helpers
pip install "easier-openai[speech_models]"    # whisper speech recognition models
```

Set the `OPENAI_API_KEY` environment variable or pass `api_key` directly when instantiating `Assistant`.

## Quick Start
```python
from easier_openai import Assistant

assistant = Assistant(model="gpt-4o-mini", system_prompt="You are concise.")
response_text = assistant.chat("Summarize Rayleigh scattering in one sentence.")
print(response_text)
```

## Tool Calling Made Simple
Use `Assistant.openai_function` (also re-exported at the package root) to convert regular functions into structured tool definitions:

```python
from easier_openai import Assistant, openai_function

assistant = Assistant()

@openai_function
def look_up_fact(topic: str) -> dict:
    """Return a knowledge base lookup result for the given topic."""
    return {"topic": topic}

assistant.chat(
    "Tell me about the ozone layer using the fact tool.",
    custom_tools=[look_up_fact],
)
```

## Ground Responses With Your Files
```python
notes = ["notes/overview.md", "notes/data-sheet.pdf"]
reply = assistant.chat(
    "Highlight key risks from the attached docs",
    file_search=notes,
    tools_required="auto",
)
print(reply)
```

## Speech I/O
Generate audio output directly from assistant responses:

```python
assistant.full_text_to_speech(
    "Ship a status update that sounds upbeat",
    model="gpt-4o-mini-tts",
    voice="alloy",
    play=True,
)
```

Or capture short dictated prompts without leaving the terminal:

```python
transcript = assistant.speech_to_text(mode="vad", model="base.en")
print(transcript)
```

## Image Utilities
`Openai_Images` extends the assistant with helpers that accept URLs, file paths, or base64 payloads and normalise them for the Images API:

```python
from easier_openai import Openai_Images

image_client = Openai_Images("samples/promenade.jpg")
# Generated metadata is stored on image_client.image for re-use in calls.
```

## Configuration Reference
- `model`: Default model used for chat, tool calls, and reasoning workflows.
- `system_prompt`: Injected once per conversation to shape assistant behaviour.
- `reasoning_effort` and `summary_length`: Fine tune reasoning models via the official API semantics.
- `temperature`: Pass through value mapped to OpenAI responses for deterministic vs creative answers.
- `default_conversation`: Set to `False` if you prefer to supply conversation IDs manually.
- `mass_update(**kwargs)`: Bulk update configuration attributes using keyword arguments validated by type hints.

## Developer Notes
- Every public function and class ships with contextual docstrings to make the codebase self-documenting.
- The repository includes unit tests under `tests/` that exercise tool-calling flows; run them with `pytest`.
- Generated artifacts in `build/` mirror the source package and inherit the same documentation updates.
- Issues and pull requests are welcome; please run checks locally before submitting changes.

## License
Licensed under the [Apache License 2.0](LICENSE).
