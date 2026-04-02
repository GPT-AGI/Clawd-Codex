"""Tests for chinese_models module."""

import unittest

from src.chinese_models import (
    ChineseModelInfo,
    ChineseModelRegistry,
    ChineseModelAdapter,
    create_chinese_model_registry,
)


class TestChineseModels(unittest.TestCase):
    """Test cases for Chinese models."""

    def test_create_chinese_model_info(self):
        """Test creating model info."""
        model = ChineseModelInfo(
            provider="test",
            provider_cn="测试",
            model="test-model",
            model_cn="测试模型",
            description="Test description",
            api_endpoint="https://api.test.com",
            max_tokens=4096,
        )

        self.assertEqual(model.provider, "test")
        self.assertEqual(model.model_cn, "测试模型")

    def test_create_registry(self):
        """Test creating registry."""
        registry = create_chinese_model_registry()

        self.assertIsInstance(registry, ChineseModelRegistry)
        self.assertGreater(len(registry.models), 0)

    def test_get_model(self):
        """Test getting a model."""
        registry = ChineseModelRegistry()
        model = registry.get_model("qwen-max")

        self.assertIsNotNone(model)
        self.assertEqual(model.provider, "alibaba")

    def test_get_model_missing(self):
        """Test getting missing model."""
        registry = ChineseModelRegistry()
        model = registry.get_model("nonexistent")

        self.assertIsNone(model)

    def test_list_models(self):
        """Test listing all models."""
        registry = ChineseModelRegistry()
        models = registry.list_models()

        self.assertGreater(len(models), 0)

    def test_list_by_provider(self):
        """Test listing models by provider."""
        registry = ChineseModelRegistry()
        models = registry.list_by_provider("alibaba")

        self.assertGreater(len(models), 0)
        for model in models:
            self.assertEqual(model.provider, "alibaba")

    def test_get_providers(self):
        """Test getting all providers."""
        registry = ChineseModelRegistry()
        providers = registry.get_providers()

        self.assertIn("alibaba", providers)
        self.assertIn("zhipu", providers)

    def test_search_models(self):
        """Test searching models."""
        registry = ChineseModelRegistry()
        results = registry.search_models("qwen")

        self.assertGreater(len(results), 0)

    def test_search_models_chinese(self):
        """Test searching with Chinese characters."""
        registry = ChineseModelRegistry()
        results = registry.search_models("通义")

        self.assertGreater(len(results), 0)

    def test_adapter_format_request(self):
        """Test formatting request."""
        model = ChineseModelInfo(
            provider="test",
            provider_cn="测试",
            model="test-model",
            model_cn="测试模型",
            description="Test",
            api_endpoint="https://api.test.com",
            max_tokens=4096,
        )

        messages = [{"role": "user", "content": "Hello"}]
        request = ChineseModelAdapter.format_request(model, messages)

        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["messages"], messages)

    def test_adapter_format_request_with_params(self):
        """Test formatting request with parameters."""
        model = ChineseModelInfo(
            provider="test",
            provider_cn="测试",
            model="test-model",
            model_cn="测试模型",
            description="Test",
            api_endpoint="https://api.test.com",
            max_tokens=4096,
        )

        messages = [{"role": "user", "content": "Hello"}]
        request = ChineseModelAdapter.format_request(
            model, messages, temperature=0.7, max_tokens=100
        )

        self.assertEqual(request["temperature"], 0.7)
        self.assertEqual(request["max_tokens"], 100)

    def test_adapter_parse_response(self):
        """Test parsing response."""
        response = {
            "choices": [{"message": {"content": "Test response"}}],
            "model": "test-model",
            "usage": {"total_tokens": 10},
        }

        parsed = ChineseModelAdapter.parse_response(response)

        self.assertEqual(parsed["content"], "Test response")
        self.assertEqual(parsed["model"], "test-model")

    def test_builtin_models_available(self):
        """Test that built-in models are available."""
        registry = ChineseModelRegistry()

        # Check some key models
        self.assertIsNotNone(registry.get_model("qwen-max"))
        self.assertIsNotNone(registry.get_model("glm-4-plus"))
        self.assertIsNotNone(registry.get_model("deepseek-chat"))
        self.assertIsNotNone(registry.get_model("moonshot-v1-8k"))


if __name__ == "__main__":
    unittest.main()
