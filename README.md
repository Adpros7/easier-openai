# Easier OpenAI

Easier OpenAI wraps the official OpenAI Python SDK so you can drive modern assistants, manage tool selection, search files, and work with speech from one helper package—the easiest possible way.

## What's Included
- Conversational `Assistant` helper with conversation memory and tool toggles.
- Temporary vector store ingestion to ground answers in local notes.
- Built-in helpers for image generation and text-to-speech playback.
- Speech-to-text recording shortcuts for quick dictation.
- Lazy-loaded imports so `import easier_openai` stays fast even when optional helpers expand.
- Optional `openai_function` decorator re-exported for function tool schemas.

## Installation
```bash
pip install easier-openai
```

Optional extras:
```bash
pip install "easier-openai[function_tools]"   # decorator helpers
pip install "easier-openai[speech_models]"    # whisper speech recognition models
```

Set `OPENAI_API_KEY` in your environment or pass it explicitly when constructing the assistant.

## Quick Start
```python
from easier_openai import Assistant

assistant = Assistant(api_key=None, model="gpt-4o", system_prompt="You are concise.")
response_text = assistant.chat("Summarize Rayleigh scattering in one sentence.")
print(response_text)
```

### Ground replies with your files
```python
notes = ["notes/overview.md", "notes/data-sheet.pdf"]
reply = assistant.chat(
    "Highlight key risks from the attached docs",
    file_search=notes,
    tools_required="auto",
)
print(reply)
```

### Generate speech from responses
```python
assistant.full_text_to_speech(
    "Ship a status update that sounds upbeat",
    model="gpt-4o-mini-tts",
    voice="alloy",
    play=True,
)
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
