"""Easy GPT helper package.

This module exposes the high-level Assistant wrapper and optional
function-calling decorator helpers. The project is distributed under the
Apache License, Version 2.0. See the accompanying LICENSE document for
more information.
"""

from importlib import metadata as _metadata

from .assistant import Assistant

__all__ = ["Assistant", "__version__"]

try:
    __version__ = _metadata.version("easy-gpt")
except _metadata.PackageNotFoundError:  # Running from source tree without metadata
    __version__ = "0.dev0"