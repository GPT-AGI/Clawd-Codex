"""Configuration management for Clawd Codex."""

from __future__ import annotations

import json
import base64
from pathlib import Path
from typing import Any, Optional


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    config_dir = Path.home() / ".clawd"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def get_default_config() -> dict[str, Any]:
    """Generate default configuration."""
    return {
        "default_provider": "glm",
        "providers": {
            "anthropic": {
                "api_key": "",
                "base_url": "https://api.anthropic.com",
                "default_model": "claude-sonnet-4-20250514"
            },
            "openai": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "default_model": "gpt-4"
            },
            "glm": {
                "api_key": "",
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "default_model": "glm-4.5"
            }
        },
        "session": {
            "auto_save": True,
            "max_history": 100
        }
    }


def _encode_api_key(api_key: str) -> str:
    """Encode API key for basic obfuscation."""
    return base64.b64encode(api_key.encode()).decode()


def _decode_api_key(encoded_key: str) -> str:
    """Decode API key."""
    try:
        return base64.b64decode(encoded_key.encode()).decode()
    except Exception:
        # If decoding fails, return as-is (might be plain text)
        return encoded_key


def load_config() -> dict[str, Any]:
    """Load configuration from file.

    Returns:
        Configuration dictionary
    """
    config_path = get_config_path()

    if not config_path.exists():
        # Create default config
        config = get_default_config()
        save_config(config)
        return config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Decode API keys
        for provider_name, provider_config in config.get("providers", {}).items():
            if provider_config.get("api_key"):
                provider_config["api_key"] = _decode_api_key(provider_config["api_key"])

        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return get_default_config()


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary to save
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a copy to avoid modifying the original
    config_copy = json.loads(json.dumps(config))

    # Encode API keys
    for provider_name, provider_config in config_copy.get("providers", {}).items():
        if provider_config.get("api_key"):
            provider_config["api_key"] = _encode_api_key(provider_config["api_key"])

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_copy, f, indent=2, ensure_ascii=False)


def get_provider_config(provider: str) -> dict[str, Any]:
    """Get configuration for a specific provider.

    Args:
        provider: Provider name (anthropic, openai, glm)

    Returns:
        Provider configuration dictionary
    """
    config = load_config()
    providers = config.get("providers", {})

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")

    return providers[provider]


def set_api_key(provider: str, api_key: str, base_url: Optional[str] = None,
                default_model: Optional[str] = None) -> None:
    """Set API key for a provider.

    Args:
        provider: Provider name (anthropic, openai, glm)
        api_key: API key to set
        base_url: Optional base URL override
        default_model: Optional default model override
    """
    config = load_config()

    if provider not in config.get("providers", {}):
        # Add new provider if it doesn't exist
        if "providers" not in config:
            config["providers"] = {}
        config["providers"][provider] = {}

    config["providers"][provider]["api_key"] = api_key

    if base_url is not None:
        config["providers"][provider]["base_url"] = base_url

    if default_model is not None:
        config["providers"][provider]["default_model"] = default_model

    save_config(config)


def set_default_provider(provider: str) -> None:
    """Set the default provider.

    Args:
        provider: Provider name
    """
    config = load_config()
    config["default_provider"] = provider
    save_config(config)


def get_default_provider() -> str:
    """Get the default provider.

    Returns:
        Default provider name
    """
    config = load_config()
    return config.get("default_provider", "glm")
