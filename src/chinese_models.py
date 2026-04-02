"""
Chinese AI model ecosystem first-class support.

Provides enhanced support for Chinese AI models:
- Baidu (Wenxin Yiyan)
- Alibaba (Qwen/Tongyi Qianwen)
- Tencent (Hunyuan)
- iFlytek (Spark)
- Baichuan
- Moonshot (Kimi)
- DeepSeek
- MiniMax
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChineseModelInfo:
    """Chinese model information."""

    provider: str
    provider_cn: str  # Chinese name
    model: str
    model_cn: str  # Chinese name
    description: str
    api_endpoint: str
    max_tokens: int
    supports_streaming: bool = True
    supports_tools: bool = False
    pricing_rmb_per_1k_tokens: float = 0.0


# Chinese model registry
CHINESE_MODELS = {
    # Baidu Wenxin
    "ernie-4.0": ChineseModelInfo(
        provider="baidu",
        provider_cn="百度",
        model="ernie-4.0",
        model_cn="文心一言4.0",
        description="Baidu's most advanced model",
        api_endpoint="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
        max_tokens=8000,
        pricing_rmb_per_1k_tokens=0.12,
    ),
    # Alibaba Qwen
    "qwen-max": ChineseModelInfo(
        provider="alibaba",
        provider_cn="阿里云",
        model="qwen-max",
        model_cn="通义千问-Max",
        description="Alibaba's most powerful model",
        api_endpoint="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        max_tokens=6000,
        supports_tools=True,
        pricing_rmb_per_1k_tokens=0.12,
    ),
    "qwen-turbo": ChineseModelInfo(
        provider="alibaba",
        provider_cn="阿里云",
        model="qwen-turbo",
        model_cn="通义千问-Turbo",
        description="Fast and cost-effective",
        api_endpoint="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        max_tokens=6000,
        pricing_rmb_per_1k_tokens=0.008,
    ),
    # Zhipu GLM (already in providers, but enhance here)
    "glm-4-plus": ChineseModelInfo(
        provider="zhipu",
        provider_cn="智谱AI",
        model="glm-4-plus",
        model_cn="GLM-4增强版",
        description="Zhipu's most advanced model",
        api_endpoint="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        max_tokens=128000,
        supports_tools=True,
        pricing_rmb_per_1k_tokens=0.05,
    ),
    # Baichuan
    "baichuan2-turbo": ChineseModelInfo(
        provider="baichuan",
        provider_cn="百川智能",
        model="baichuan2-turbo",
        model_cn="百川2-Turbo",
        description="Fast and affordable",
        api_endpoint="https://api.baichuan-ai.com/v1/chat/completions",
        max_tokens=4096,
        pricing_rmb_per_1k_tokens=0.008,
    ),
    # Moonshot Kimi
    "moonshot-v1-8k": ChineseModelInfo(
        provider="moonshot",
        provider_cn="月之暗面",
        model="moonshot-v1-8k",
        model_cn="Kimi",
        description="Long context specialist",
        api_endpoint="https://api.moonshot.cn/v1/chat/completions",
        max_tokens=8192,
        pricing_rmb_per_1k_tokens=0.012,
    ),
    "moonshot-v1-32k": ChineseModelInfo(
        provider="moonshot",
        provider_cn="月之暗面",
        model="moonshot-v1-32k",
        model_cn="Kimi长文本",
        description="32K context window",
        api_endpoint="https://api.moonshot.cn/v1/chat/completions",
        max_tokens=32768,
        pricing_rmb_per_1k_tokens=0.024,
    ),
    # DeepSeek
    "deepseek-chat": ChineseModelInfo(
        provider="deepseek",
        provider_cn="深度求索",
        model="deepseek-chat",
        model_cn="DeepSeek对话",
        description="High quality at low cost",
        api_endpoint="https://api.deepseek.com/v1/chat/completions",
        max_tokens=16000,
        supports_tools=True,
        pricing_rmb_per_1k_tokens=0.001,  # Very cheap
    ),
    "deepseek-coder": ChineseModelInfo(
        provider="deepseek",
        provider_cn="深度求索",
        model="deepseek-coder",
        model_cn="DeepSeek编程",
        description="Code specialist",
        api_endpoint="https://api.deepseek.com/v1/chat/completions",
        max_tokens=16000,
        supports_tools=True,
        pricing_rmb_per_1k_tokens=0.001,
    ),
    # MiniMax
    "abab5.5-chat": ChineseModelInfo(
        provider="minimax",
        provider_cn="MiniMax",
        model="abab5.5-chat",
        model_cn="ABAB 5.5",
        description="MiniMax flagship model",
        api_endpoint="https://api.minimax.chat/v1/text/chatcompletion_v2",
        max_tokens=6144,
        supports_tools=True,
        pricing_rmb_per_1k_tokens=0.015,
    ),
}


class ChineseModelRegistry:
    """Registry for Chinese AI models."""

    def __init__(self):
        """Initialize registry."""
        self.models = CHINESE_MODELS.copy()

    def get_model(self, model_id: str) -> ChineseModelInfo | None:
        """Get model information."""
        return self.models.get(model_id)

    def list_models(self) -> list[ChineseModelInfo]:
        """List all registered models."""
        return list(self.models.values())

    def list_by_provider(self, provider: str) -> list[ChineseModelInfo]:
        """List models by provider."""
        return [m for m in self.models.values() if m.provider == provider]

    def get_providers(self) -> list[str]:
        """Get all providers."""
        return list(set(m.provider for m in self.models.values()))

    def search_models(self, query: str) -> list[ChineseModelInfo]:
        """Search models by name or description."""
        query = query.lower()
        results = []

        for model in self.models.values():
            if (
                query in model.model.lower()
                or query in model.model_cn
                or query in model.description.lower()
                or query in model.provider.lower()
                or query in model.provider_cn
            ):
                results.append(model)

        return results


class ChineseModelAdapter:
    """Adapts Chinese models to common interface."""

    @staticmethod
    def format_request(
        model: ChineseModelInfo,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> dict[str, Any]:
        """
        Format request for Chinese model API.

        Args:
            model: Model information
            messages: Chat messages
            **kwargs: Additional parameters

        Returns:
            Formatted request
        """
        # Most Chinese models follow OpenAI-compatible API
        request = {
            "model": model.model,
            "messages": messages,
        }

        # Add optional parameters
        if "temperature" in kwargs:
            request["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            request["max_tokens"] = min(kwargs["max_tokens"], model.max_tokens)
        if "stream" in kwargs:
            request["stream"] = kwargs["stream"]

        return request

    @staticmethod
    def parse_response(response: dict[str, Any]) -> dict[str, Any]:
        """
        Parse response from Chinese model API.

        Args:
            response: API response

        Returns:
            Parsed response
        """
        # Most use OpenAI-compatible format
        return {
            "content": response.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "model": response.get("model"),
            "usage": response.get("usage", {}),
        }


def create_chinese_model_registry() -> ChineseModelRegistry:
    """Create a Chinese model registry."""
    return ChineseModelRegistry()
