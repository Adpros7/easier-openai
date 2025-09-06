from __future__ import annotations

import gc
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
        """Change the system prompt for future calls."""

        self.system_prompt = new_prompt

    def add_tool(self, tool: Dict[str, Any]) -> None:
        """Register a tool for function calling."""

        self.tools.append(tool)

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------
    def ask(self, prompt: str) -> str:
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


__all__ = ["Assistant"]

