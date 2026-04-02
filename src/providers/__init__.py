"""LLM Providers for Clawd Codex."""

from __future__ import annotations

from .base import BaseProvider, ChatMessage, ChatResponse


def get_provider_class(provider_name: str):
    """Get provider class by name."""
    if provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider
    if provider_name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider
    if provider_name == "glm":
        from .glm_provider import GLMProvider

        return GLMProvider
    raise ValueError(f"Unknown provider: {provider_name}")


__all__ = [
    "BaseProvider",
    "ChatMessage",
    "ChatResponse",
    "get_provider_class",
]
