from __future__ import annotations

import gc

from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Literal, Optional

from openai import AsyncOpenAI, OpenAI

# Reuse clients to avoid repeated HTTP setup overhead
_SYNC_CLIENT: OpenAI = OpenAI()
_ASYNC_CLIENT: AsyncOpenAI = AsyncOpenAI()

# Commonly used OpenAI responses models for IDE auto-complete
ModelName = Literal["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]

"""Literal type for the most common OpenAI model names."""

# Conversation roles and context store
Role = Literal["user", "assistant"]
"""Allowed participant roles in the conversation history."""

Context = Dict[Role, List[str]]
"""Simple mapping of roles to their ordered utterances."""




@dataclass(slots=True)
class Assistant:

    """Wrapper around the OpenAI Responses API.

    Parameters
    ----------
    system_prompt:
        Initial system prompt guiding the assistant's behaviour.
    model:
        Model name to use. Defaults to ``"gpt-4o-mini"`` but any model
        identifier may be supplied.
    api_key:
        Optional OpenAI API key. Falls back to ``OPENAI_API_KEY``
        from the environment when not provided.
    tools:
        Optional collection of JSON schemas describing callable tools.
    use_context:
        Track conversation history for later reuse when ``True``.
    """

    """Wrapper around the OpenAI Responses API."""


    system_prompt: str
    model: ModelName | str = "gpt-4o-mini"
    api_key: Optional[str] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    use_context: bool = False

    client: OpenAI = field(init=False)
    async_client: AsyncOpenAI = field(init=False)

    context: Optional[Context] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize OpenAI clients and optional context store."""


    context: Optional[Dict[str, List[str]]] = field(init=False)

    def __post_init__(self) -> None:

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = _SYNC_CLIENT
            self.async_client = _ASYNC_CLIENT
        self.context = {"user": [], "assistant": []} if self.use_context else None

from typing import Any, Dict, Iterable, List, Optional

from openai import AsyncOpenAI, OpenAI


class Assistant:
    """Wrapper around the OpenAI Responses API.

    Parameters
    ----------
    system_prompt:
        The system prompt that guides the assistant.
    model:
        Model name to use for responses. Any valid model string is accepted.
    api_key:
        API key for OpenAI. If ``None`` the environment variable
        ``OPENAI_API_KEY`` is used.
    tools:
        List of tool specifications for function calling.
    use_context:
        When ``True`` the assistant stores message history that can be
        accessed via ``.context``.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        use_context: bool = False,
    ) -> None:
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.async_client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.use_context = use_context
        self.context: Optional[Dict[str, List[str]]] = (
            {"user": [], "assistant": []} if use_context else None
        )



    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def update_system_prompt(self, new_prompt: str) -> None:

        """Change the system prompt for future calls.

        Parameters
        ----------
        new_prompt:
            Replacement system instruction for subsequent responses.
        """

        """Change the system prompt for future calls."""


        self.system_prompt = new_prompt

    def add_tool(self, tool: Dict[str, Any]) -> None:

        """Register a tool for function calling.

        Parameters
        ----------
        tool:
            JSON schema describing the callable tool.
        """

        """Register a tool for function calling."""


        self.tools.append(tool)

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------
    def ask(self, prompt: str) -> str:

        """Return a response synchronously.

        Parameters
        ----------
        prompt:
            User message to send to the model.

        Returns
        -------
        str
            Response text from the model.
        """

        """Return a response synchronously."""


        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "instructions": self.system_prompt,
        }
        if self.tools:
            kwargs["tools"] = self.tools

        response = self.client.responses.create(**kwargs)
        text = str(response.output_text)

        if self.use_context and self.context is not None:
            self.context["user"].append(prompt)
            self.context["assistant"].append(text)

        return text


    def ask_stream(self, prompt: str) -> Generator[str, None, None]:
        """Yield partial responses as they stream in.

        Parameters
        ----------
        prompt:
            User message to send to the model.

        Yields
        ------
        str
            Growing partial response text.
        """


    def ask_stream(self, prompt: str) -> Generator[str, None, None]:

    def ask_stream(self, prompt: str) -> Iterable[str]:

        """Yield partial responses as they stream in."""


        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "instructions": self.system_prompt,
            "stream": True,
        }
        if self.tools:
            kwargs["tools"] = self.tools

        response = self.client.responses.create(**kwargs)
        partial = ""
        for chunk in response:
            if hasattr(chunk, "output_text"):
                partial += str(chunk.output_text)
                yield partial

        if self.use_context and self.context is not None:
            self.context["user"].append(prompt)
            self.context["assistant"].append(partial)

    # ------------------------------------------------------------------
    # Asynchronous API
    # ------------------------------------------------------------------
    async def ask_async(self, prompt: str) -> str:

        """Return a response using ``AsyncOpenAI``.

        Parameters
        ----------
        prompt:
            User message to send to the model.

        Returns
        -------
        str
            Response text from the model.
        """

        """Return a response using ``AsyncOpenAI``."""


        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "instructions": self.system_prompt,
        }
        if self.tools:
            kwargs["tools"] = self.tools

        response = await self.async_client.responses.create(**kwargs)
        text = str(response.output_text)

        if self.use_context and self.context is not None:
            self.context["user"].append(prompt)
            self.context["assistant"].append(text)

        return text


    async def ask_stream_async(self, prompt: str) -> AsyncGenerator[str, None]:
        """Asynchronously yield partial responses.

        Parameters
        ----------
        prompt:
            User message to send to the model.

        Yields
        ------
        str
            Growing partial response text.
        """


    async def ask_stream_async(self, prompt: str) -> AsyncGenerator[str, None]:

    async def ask_stream_async(self, prompt: str) -> Iterable[str]:

        """Asynchronously yield partial responses."""


        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "instructions": self.system_prompt,
            "stream": True,
        }
        if self.tools:
            kwargs["tools"] = self.tools

        stream = await self.async_client.responses.create(**kwargs)
        partial = ""
        async for chunk in stream:
            if hasattr(chunk, "output_text"):
                partial += str(chunk.output_text)
                yield partial

        if self.use_context and self.context is not None:
            self.context["user"].append(prompt)
            self.context["assistant"].append(partial)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def clear_context(self) -> None:
        """Reset stored context and run the garbage collector."""

        if self.context is not None:
            self.context = {"user": [], "assistant": []}
        gc.collect()



__all__ = ["Assistant", "ModelName", "Role", "Context"]


__all__ = ["Assistant", "ModelName"]

__all__ = ["Assistant"]



