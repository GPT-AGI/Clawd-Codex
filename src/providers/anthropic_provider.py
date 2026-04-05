"""Anthropic provider implementation."""

from __future__ import annotations

from typing import Generator, Optional, Any

try:
    import anthropic  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _MissingAnthropic:
        class Anthropic:  # type: ignore[no-redef]
            def __init__(self, *args: list, **kwargs: dict):
                raise ModuleNotFoundError(
                    "anthropic package is not installed. Install optional dependencies to use AnthropicProvider."
                )

    anthropic = _MissingAnthropic()

from .base import BaseProvider, ChatResponse, MessageInput


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider."""

    def __init__(
        self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            base_url: Base URL (optional)
            model: Default model (default: claude-sonnet-4-6)
        """
        super().__init__(api_key, base_url, model or "claude-sonnet-4-6")

        self._client_kwargs = {"api_key": api_key}
        if base_url:
            self._client_kwargs["base_url"] = base_url
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
            **kwargs: Additional parameters (model, max_tokens, temperature, etc.)

        Returns:
            Chat response
        """
        model = self._get_model(**kwargs)
        max_tokens = kwargs.get("max_tokens", 4096)

        system = kwargs.pop("system", None)

        # Convert messages to Anthropic format
        anthropic_messages = self._prepare_messages(messages)

        # Make API call
        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=anthropic_messages,
            **({"system": system} if system else {}),
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
        )

        # Extract content and tool uses
        content_text = ""
        tool_uses: list[dict[str, Any]] = []

        for block in response.content:
            # Handle both real Anthropic blocks and test mocks
            block_type = getattr(block, "type", "text")
            if block_type == "text":
                # Get text safely for both real blocks and mocks
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
        anthropic_messages = self._prepare_messages(messages)

        # Stream API call
        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=anthropic_messages,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
        ) as stream:
            for text in stream.text_stream:
                yield text

    def get_available_models(self) -> list[str]:
        """Get list of available Anthropic models.

        Returns:
            List of model names
        """
        return [
            # Claude 4 series (latest)
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-0",
            "claude-sonnet-4-20250514",
            "claude-opus-4-6",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-opus-4-1",
            "claude-opus-4-1-20250805",
            "claude-opus-4-0",
            "claude-opus-4-20250514",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            # Legacy
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
