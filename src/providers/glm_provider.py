"""GLM (Zhipu AI) provider implementation."""

from __future__ import annotations

from typing import Generator, Optional, Any

try:
    from zhipuai import ZhipuAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ZhipuAI = None

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


class GLMProvider(BaseProvider):
    """GLM (Zhipu AI) provider."""

    def __init__(
        self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        """Initialize GLM provider.

        Args:
            api_key: Zhipu AI API key
            base_url: Base URL (optional)
            model: Default model (default: glm-4.5)
        """
        super().__init__(api_key, base_url, model or "glm-4.5")

        self._api_key = api_key
        self.client = None

    def _ensure_client(self):
        if self.client is not None:
            return self.client
        if ZhipuAI is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "zhipuai package is not installed. Install optional dependencies to use GLMProvider."
            )
        self.client = ZhipuAI(api_key=self._api_key)
        return self.client

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """Synchronous chat completion.

        Args:
            messages: List of chat messages (ChatMessage or dict)
            tools: Optional list of tool schemas (Anthropic format)
            **kwargs: Additional parameters

        Returns:
            Chat response
        """
        model = self._get_model(**kwargs)

        # Convert messages
        glm_messages = self._prepare_messages(messages)

        # Convert tools to OpenAI/GLM format
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = [_convert_to_openai_tool_schema(t) for t in tools]

        # Make API call
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=model,
            messages=glm_messages,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "tools"]},
        )

        # Extract content
        choice = response.choices[0]

        # GLM-4.5 specific: reasoning_content
        reasoning_content = None
        if (
            hasattr(choice.message, "reasoning_content")
            and choice.message.reasoning_content
        ):
            reasoning_content = choice.message.reasoning_content

        # Extract tool calls (OpenAI/GLM format -> Anthropic format)
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
            reasoning_content=reasoning_content,
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
            messages: List of chat messages (ChatMessage or dict)
            tools: Optional list of tool schemas (Anthropic format)
            **kwargs: Additional parameters

        Yields:
            Chunks of response content
        """
        model = self._get_model(**kwargs)

        # Convert messages
        glm_messages = self._prepare_messages(messages)

        # Convert tools to OpenAI/GLM format
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = [_convert_to_openai_tool_schema(t) for t in tools]

        # Stream API call
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=model,
            messages=glm_messages,
            stream=True,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "tools"]},
        )

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_available_models(self) -> list[str]:
        """Get list of available GLM models.

        Returns:
            List of model names
        """
        return [
            "glm-4.5",
            "glm-4",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-flash",
            "glm-4-plus",
            "glm-3-turbo",
        ]
