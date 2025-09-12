"""Easy-gPT: Lightweight helpers for OpenAI's Python SDK.

This package exposes a thin, convenient wrapper around the OpenAI
Responses API, Conversations, Vector Stores (for file search), and
Images generation via the `Assistant` class.

Quick start (requires OPENAI_API_KEY set in your environment):

    from importlib import import_module
    assistant_mod = import_module('Easy-gpt.assistant')
    Assistant = assistant_mod.Assistant
    bot = Assistant(api_key=None, model="gpt-4o", system_prompt="You are helpful.")
    text, conversation = bot.chat("Hello!")

Note about imports:
- The package directory is named with a hyphen ("Easy-gpt"). The standard
  Python `import` statement does not allow hyphens in identifiers. Use
  `importlib.import_module('Easy-gpt.assistant')` as shown above, or
  import the file relatively from your own package/module. The internal
  relative imports below are valid within this package itself.
"""

from .assistant import Assistant, CustomToolInputFormat

__all__ = ["Assistant", "CustomToolInputFormat"]
__version__ = "0.1.0"
