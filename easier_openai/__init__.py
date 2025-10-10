"""One-stop helpers for coordinating OpenAI chat, tools, retrieval, and media flows."""

from importlib import import_module
from importlib import metadata as _metadata
from typing import TYPE_CHECKING, Any, List

from .assistant import preload_openai_stt

__all__ = [
    "Assistant",
    "__version__",
    "__description__",
    "Seconds",
    "VadAgressiveness",
    "Openai_Images",
    "preload_openai_stt",
]

_DISTRIBUTION_NAME = "easier-openai"

_LAZY_EXPORTS = {
    "Assistant": ("easier_openai.assistant", "Assistant"),
    "Seconds": ("easier_openai.assistant", "Seconds"),
    "VadAgressiveness": ("easier_openai.assistant", "VadAgressiveness"),
    "Openai_Images": ("easier_openai.Images", "Openai_Images"),
}

if TYPE_CHECKING:
    from .assistant import Assistant, Seconds, VadAgressiveness
    from .Images import Openai_Images


def __getattr__(name: str) -> Any:
    """Load heavy modules lazily when their exports are first accessed.

    Example:
        >>> from easier_openai import Assistant  # triggers __getattr__
        >>> isinstance(Assistant, type)
        True

    Note:
        Resolved attributes are memoised on the module so subsequent imports are immediate.
    """
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:  # pragma: no cover - defers to Python's default error
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """Expose lazy exports when introspecting the module via dir().

    Example:
        >>> \"Assistant\" in __dir__()
        True
    """
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
