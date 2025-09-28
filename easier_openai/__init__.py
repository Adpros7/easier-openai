"""One-stop helpers for coordinating OpenAI chat, tools, retrieval, and media flows."""

from importlib import metadata as _metadata

from .assistant import Assistant  # re-export primary helper
from .assistant import Seconds
from ez_openai.decorator import openai_function  # convenience decorator re-export

__all__ = ["Assistant", "openai_function", "__version__", "__description__", "Seconds"]

_DISTRIBUTION_NAME = "easier-openai"

try:
    __version__ = _metadata.version(_DISTRIBUTION_NAME)
except _metadata.PackageNotFoundError:  # Running from a source tree without installed metadata
    __version__ = "0.3.0"

try:
    __description__ = _metadata.metadata(_DISTRIBUTION_NAME)["Summary"]
except (KeyError, _metadata.PackageNotFoundError):
    __description__ = (
        "Utilities for orchestrating OpenAI chat, tool calling, search, "
        "audio, and images from one helper package."
    )
