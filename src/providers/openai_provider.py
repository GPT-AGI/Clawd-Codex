"""OpenAI provider implementation."""

from __future__ import annotations

from typing import Any, Generator, Optional

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None

from .base import BaseProvider, ChatResponse, MessageInput


def _convert_to_openai_tool_schema(anthropic_tool: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic tool schema to OpenAI/GLM format."""
    return {
        "type": "function",
        "function": {
            "name": anthropic_tool["name"],
            "description": anthropic_tool.get("description", ""),
            "parameters": anthropic_tool.get("input_schema", {}),
        },
    }


class OpenAIProvider(BaseProvider):
    """OpenAI provider."""

    def __init__(
        self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: Base URL (optional, for custom endpoints)
            model: Default model (default: gpt-4)
        """
        super().__init__(api_key, base_url, model or "gpt-4")

        self._client_kwargs = {"api_key": api_key}
        if base_url:
            self._client_kwargs["base_url"] = base_url
        self.client = None

    def _ensure_client(self):
        if self.client is not None:
            return self.client
        if OpenAI is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "openai package is not installed. Install optional dependencies to use OpenAIProvider."
            )
        self.client = OpenAI(**self._client_kwargs)
        return self.client

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """Synchronous chat completion.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas (Anthropic format)
            **kwargs: Additional parameters

        Returns:
            Chat response
        """
        model = self._get_model(**kwargs)

        # Convert messages
        openai_messages = self._prepare_messages(messages)

        # Convert tools to OpenAI format
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = [_convert_to_openai_tool_schema(t) for t in tools]

        # Make API call
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "tools"]},
        )

        # Extract content
        choice = response.choices[0]

        # Extract tool calls (OpenAI format -> Anthropic format)
        tool_uses: Optional[list[dict[str, Any]]] = None
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_uses = []
            for tc in choice.message.tool_calls:
                import json
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except Exception:
                    args = {}
                tool_uses.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                })

        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
            tool_uses=tool_uses,
        )

    def chat_stream(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Streaming chat completion.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas (Anthropic format)
            **kwargs: Additional parameters

        Yields:
            Chunks of response content
        """
        model = self._get_model(**kwargs)

        # Convert messages
        openai_messages = self._prepare_messages(messages)

        # Convert tools to OpenAI format
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = [_convert_to_openai_tool_schema(t) for t in tools]

        # Stream API call
        client = self._ensure_client()
        stream = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            stream=True,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "tools"]},
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_available_models(self) -> list[str]:
        """Get list of available OpenAI models.

        Returns:
            List of model names
        """
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
