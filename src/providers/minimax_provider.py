"""Minimax provider implementation using Anthropic-compatible API."""

from __future__ import annotations

from typing import Generator, Optional, Any

try:
    import anthropic  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _MissingAnthropic:
        class Anthropic:  # type: ignore[no-redef]
            def __init__(self, *args: list, **kwargs: dict):
                raise ModuleNotFoundError(
                    "anthropic package is not installed. Install optional dependencies to use MinimaxProvider."
                )

    anthropic = _MissingAnthropic()

from .base import BaseProvider, ChatResponse, MessageInput


class MinimaxProvider(BaseProvider):
    """Minimax AI provider using Anthropic-compatible API.

    Minimax provides an Anthropic-compatible endpoint at api.minimaxi.com/anthropic.
    Uses the Anthropic SDK with Minimax-specific models.
    """

    DEFAULT_BASE_URL = "https://api.minimaxi.com/anthropic"

    def __init__(
        self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        """Initialize Minimax provider.

        Args:
            api_key: Minimax API key
            base_url: Base URL (optional, defaults to Minimax Anthropic-compatible endpoint)
            model: Default model (default: MiniMax-M2.7)
        """
        resolved_base_url = base_url or self.DEFAULT_BASE_URL
        super().__init__(api_key, resolved_base_url, model or "MiniMax-M2.7")

        self._client_kwargs: dict[str, Any] = {"api_key": api_key}
        if resolved_base_url:
            self._client_kwargs["base_url"] = resolved_base_url
        self.client = None

    def _ensure_client(self):
        if self.client is not None:
            return self.client
        self.client = anthropic.Anthropic(**self._client_kwargs)
        return self.client

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: dict
    ) -> ChatResponse:
        """Synchronous chat completion.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas
            **kwargs: Additional parameters

        Returns:
            Chat response
        """
        model = self._get_model(**kwargs)
        max_tokens = kwargs.get("max_tokens", 4096)

        system = kwargs.pop("system", None)

        # Convert messages
        minimax_messages = self._prepare_messages(messages)

        # Make API call
        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=minimax_messages,
            **({"system": system} if system else {}),
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
        )

        # Extract content and tool uses
        content_text = ""
        tool_uses: list[dict[str, Any]] = []

        for block in response.content:
            block_type = getattr(block, "type", "text")
            if block_type == "text":
                text_val = getattr(block, "text", "")
                if text_val is not None:
                    content_text += str(text_val)
            elif block_type == "tool_use":
                tool_uses.append({
                    "id": str(getattr(block, "id", "")),
                    "name": str(getattr(block, "name", "")),
                    "input": dict(getattr(block, "input", {})),
                })

        return ChatResponse(
            content=content_text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            tool_uses=tool_uses if tool_uses else None,
        )

    def chat_stream(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: dict
    ) -> Generator[str, None, None]:
        """Streaming chat completion.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas
            **kwargs: Additional parameters

        Yields:
            Chunks of response content
        """
        model = self._get_model(**kwargs)
        max_tokens = kwargs.get("max_tokens", 4096)

        # Convert messages
        minimax_messages = self._prepare_messages(messages)

        # Stream API call
        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=minimax_messages,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
        ) as stream:
            for text in stream.text_stream:
                yield text

    def get_available_models(self) -> list[str]:
        """Get list of available Minimax models.

        Returns:
            List of model names
        """
        return [
            # M2 series (latest)
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
            "M2-her",
            # Historical
            "MiniMax-M2.1",
            "MiniMax-M2.1-highspeed",
            "MiniMax-M2",
        ]
