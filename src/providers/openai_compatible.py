"""OpenAI-compatible provider base class.

This base class consolidates shared logic for providers that use the
OpenAI-style /chat/completions API (OpenAI, GLM, Minimax, etc.).
"""

from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Generator, Optional

from .base import BaseProvider, ChatResponse, MessageInput


def _convert_to_openai_tool_schema(anthropic_tool: dict[str, Any]) -> dict[str, Any] | None:
    """Convert Anthropic tool schema to OpenAI/GLM/Minimax function format.

    Returns None if the schema is invalid (missing type, type is None, or other issues).
    """
    input_schema = anthropic_tool.get("input_schema")
    if not input_schema or not isinstance(input_schema, dict):
        return None
    schema_type = input_schema.get("type")
    if schema_type is None or schema_type == "None":
        return None
    # Some providers (Azure) require type=object to have properties
    if schema_type == "object" and "properties" not in input_schema and "anyOf" not in input_schema and "oneOf" not in input_schema:
        # Try to add an empty properties dict if none provided
        input_schema = {**input_schema, "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": anthropic_tool["name"],
            "description": anthropic_tool.get("description", ""),
            "parameters": input_schema,
        },
    }


class OpenAICompatibleProvider(BaseProvider):
    """Base class for providers using OpenAI-style chat completions API.

    Subclasses must implement:
    - _create_client(): Create and return the provider-specific SDK client
    - get_available_models(): Return list of available model names

    The client is created lazily on first use.
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            model: Default model to use
        """
        super().__init__(api_key, base_url, model)
        self._client: Optional[Any] = None

    @abstractmethod
    def _create_client(self) -> Any:
        """Create the provider-specific SDK client.

        Returns:
            An instance of the provider's SDK client.
        """
        pass

    @property
    def client(self) -> Any:
        """Get or create the SDK client (lazy initialization)."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: dict
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
        provider_messages = self._prepare_messages(messages)

        # Convert tools to OpenAI format
        extra_kwargs: dict[str, Any] = {}
        if tools:
            converted = [_convert_to_openai_tool_schema(t) for t in tools]
            extra_kwargs["tools"] = [t for t in converted if t is not None]

        # Make API call
        response = self.client.chat.completions.create(
            model=model,
            messages=provider_messages,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "tools"]},
        )

        # Extract content
        choice = response.choices[0]

        # Handle reasoning content (GLM specific, but harmless for others)
        reasoning_content: Optional[str] = None
        if (
            hasattr(choice.message, "reasoning_content")
            and choice.message.reasoning_content
        ):
            reasoning_content = choice.message.reasoning_content

        # Extract tool calls (OpenAI format -> Anthropic format)
        tool_uses: Optional[list[dict[str, Any]]] = None
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_uses = []
            for tc in choice.message.tool_calls:
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
            reasoning_content=reasoning_content,
            tool_uses=tool_uses,
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
            tools: Optional list of tool schemas (Anthropic format)
            **kwargs: Additional parameters

        Yields:
            Chunks of response content
        """
        model = self._get_model(**kwargs)

        # Convert messages
        provider_messages = self._prepare_messages(messages)

        # Convert tools to OpenAI format
        extra_kwargs: dict[str, Any] = {}
        if tools:
            converted = [_convert_to_openai_tool_schema(t) for t in tools]
            extra_kwargs["tools"] = [t for t in converted if t is not None]

        # Stream API call
        stream = self.client.chat.completions.create(
            model=model,
            messages=provider_messages,
            stream=True,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "tools"]},
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
