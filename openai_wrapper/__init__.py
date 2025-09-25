"""Easy GPT helper package.

This module exposes the high-level Assistant wrapper and optional
function-calling decorator helpers. The project is distributed under the
Apache License, Version 2.0. See the accompanying LICENSE document for
more information.
"""

from importlib import metadata as _metadata

from .assistant import Assistant

try:
    from ez_openai.decorator import openai_function
except ImportError:  # pragma: no cover - dependency optional at runtime
    def openai_function(func):
        raise RuntimeError(
            "The optional dependency 'ez-openai' is required to use openai_function."
        )

__all__ = ["Assistant", "openai_function", "__version__"]

try:
    __version__ = _metadata.version("easy-gpt")
except _metadata.PackageNotFoundError:  # Running from source tree without metadata
    __version__ = "0.dev0"