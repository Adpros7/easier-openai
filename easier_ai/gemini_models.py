from typing import Literal, TypeAlias

# Runtime copy of the Gemini model literals so importing this module succeeds.
# The .pyi stub already lists the full set for type checkers; we mirror it here
# to make sure production code can rely on the alias at runtime.
GeminiModels: TypeAlias = Literal[
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]
