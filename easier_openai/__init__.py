"""One-stop helpers for coordinating OpenAI chat, tools, retrieval, and media flows."""

from importlib import import_module
from importlib import metadata as _metadata
from typing import TYPE_CHECKING, Any, List

__all__ = [
    "Assistant",
    "openai_function",
    "__version__",
    "__description__",
    "Seconds",
    "VadAgressiveness",
    "Openai_Images",
]

_DISTRIBUTION_NAME = "easier-openai"

_LAZY_EXPORTS = {
    "Assistant": ("easier_openai.assistant", "Assistant"),
    "Seconds": ("easier_openai.assistant", "Seconds"),
    "VadAgressiveness": ("easier_openai.assistant", "VadAgressiveness"),
    "Openai_Images": ("easier_openai.Images", "Openai_Images"),
    "openai_function": ("ez_openai.decorator", "openai_function"),
}

if TYPE_CHECKING:
    from .assistant import Assistant, Seconds, VadAgressiveness
    from .Images import Openai_Images
    from ez_openai.decorator import openai_function


def __getattr__(name: str) -> Any:
    """Load heavy modules lazily when their exports are first accessed."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:  # pragma: no cover - defers to Python's default error
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


try:
    __version__ = _metadata.version(_DISTRIBUTION_NAME)
except _metadata.PackageNotFoundError:
    # Running from a source tree without installed metadata
    __version__ = "0.3.0"

try:
    __description__ = _metadata.metadata(_DISTRIBUTION_NAME)["Summary"]
except (KeyError, _metadata.PackageNotFoundError):
    __description__ = (
        "Utilities for orchestrating OpenAI chat, tool calling, search, "
        "audio, and images from one helper package."
    )
