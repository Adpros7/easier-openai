
"""Public package interface for Easy-GPT.

Exposes the :class:`~easy_gpt.assistant.Assistant` helper along with literal
type aliases for common models and conversation roles.
"""

from .assistant import Assistant, Context, ModelName, Role

__all__: list[str] = ["Assistant", "ModelName", "Role", "Context"]
=======
"""Public package interface for Easy-GPT."""


from .assistant import Assistant, ModelName

__all__ = ["Assistant", "ModelName"]
=======
from .assistant import Assistant

__all__ = ["Assistant"]



