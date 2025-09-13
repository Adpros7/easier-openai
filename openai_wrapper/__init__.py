"""Easy‑GPT: Lightweight helpers for OpenAI's Python SDK.

This package provides a convenient wrapper around OpenAI's Responses API,
Conversations, Vector Stores (for file search), and Image generation via
the `Assistant` class.

Quick start (requires `OPENAI_API_KEY` in your environment):

    from openai_wrapper import Assistant
    bot = Assistant(api_key=None, model="gpt-4o", system_prompt="You are helpful.")
    reply = bot.chat("Hello!")
    print(reply)
"""

from .assistant import Assistant, CustomToolInputFormat

__all__ = ["Assistant", "CustomToolInputFormat"]

# Expose package version from installed distribution metadata (easy-gpt)
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError  # Python 3.10+
except Exception:  # pragma: no cover
    _pkg_version = None  # type: ignore
    PackageNotFoundError = Exception  # type: ignore

try:  # Try to read the version of the installed distribution
    __version__ = _pkg_version("easy-gpt") if _pkg_version else "0.0.0"
except PackageNotFoundError:  # Not installed (e.g., running from source without metadata)
    __version__ = "0.0.0"
