"""Tests for context_injector module."""

import unittest
from pathlib import Path

from src.context_injector import ContextElement, ContextInjector
from src.context import build_port_context


class TestContextInjector(unittest.TestCase):
    """Test cases for context injection."""

    def test_create_context_element(self):
        """Test creating a context element."""
        element = ContextElement(
            content="Test content",
            source="test",
            priority=50,
            token_estimate=10,
        )

        self.assertEqual(element.content, "Test content")
        self.assertEqual(element.priority, 50)

    def test_create_injector(self):
        """Test creating a context injector."""
        injector = ContextInjector(max_context_tokens=4000)
        self.assertEqual(injector.max_context_tokens, 4000)

    def test_estimate_tokens(self):
        """Test token estimation."""
        injector = ContextInjector()

        # Empty string
        self.assertEqual(injector.estimate_tokens(""), 0)

        # Short string
        text = "Hello world"
        estimate = injector.estimate_tokens(text)
        self.assertGreater(estimate, 0)

        # Longer text should have more tokens
        longer_text = "Hello world " * 100
        longer_estimate = injector.estimate_tokens(longer_text)
        self.assertGreater(longer_estimate, estimate)

    def test_build_git_context(self):
        """Test building git context."""
        injector = ContextInjector()
        context = build_port_context(include_git=True, include_tree=False)

        element = injector.build_git_context(context.git_status)

        # Should return a ContextElement
        self.assertIsNotNone(element)
        self.assertEqual(element.source, "git_status")
        self.assertIn("Git Branch:", element.content)

    def test_build_file_tree_context(self):
        """Test building file tree context."""
        injector = ContextInjector()
        context = build_port_context(include_git=False, include_tree=True)

        element = injector.build_file_tree_context(context.file_tree)

        # Should return a ContextElement
        self.assertIsNotNone(element)
        self.assertEqual(element.source, "file_tree")
        self.assertIn("## Project Structure", element.content)

    def test_build_readme_context(self):
        """Test building README context."""
        injector = ContextInjector()
        workspace_root = Path(__file__).resolve().parent.parent

        element = injector.build_readme_context(workspace_root)

        # Should return a ContextElement
        self.assertIsNotNone(element)
        self.assertEqual(element.source, "readme")
        self.assertIn("Project Documentation", element.content)

    def test_build_claudemd_context(self):
        """Test building CLAUDE.md context."""
        injector = ContextInjector()
        workspace_root = Path(__file__).resolve().parent.parent

        element = injector.build_claudemd_context(workspace_root)

        # Should return a ContextElement
        self.assertIsNotNone(element)
        self.assertEqual(element.source, "claude_md")
        self.assertIn("Project Instructions", element.content)

    def test_build_workspace_context(self):
        """Test building complete workspace context."""
        injector = ContextInjector()
        context = build_port_context(include_git=True, include_tree=True)

        elements = injector.build_workspace_context(context)

        # Should return multiple elements
        self.assertGreater(len(elements), 0)

        # Should be sorted by priority (descending)
        priorities = [e.priority for e in elements]
        self.assertEqual(priorities, sorted(priorities, reverse=True))

    def test_trim_context(self):
        """Test context trimming."""
        injector = ContextInjector(max_context_tokens=1000)

        # Create elements that exceed limit
        elements = [
            ContextElement(
                content="A" * 4000,  # ~1000 tokens
                source="high",
                priority=100,
                token_estimate=1000,
            ),
            ContextElement(
                content="B" * 4000,
                source="medium",
                priority=50,
                token_estimate=1000,
            ),
            ContextElement(
                content="C" * 4000,
                source="low",
                priority=10,
                token_estimate=1000,
            ),
        ]

        trimmed = injector.trim_context(elements, max_tokens=1500)

        # Should keep only high priority elements
        self.assertEqual(len(trimmed), 1)
        self.assertEqual(trimmed[0].source, "high")

    def test_trim_context_preserves_priority(self):
        """Test that trimming preserves highest priority elements."""
        injector = ContextInjector()

        # Elements already sorted by priority (descending)
        elements = [
            ContextElement(content="High", source="high", priority=100, token_estimate=100),
            ContextElement(content="Medium", source="medium", priority=50, token_estimate=100),
            ContextElement(content="Low", source="low", priority=10, token_estimate=100),
        ]

        trimmed = injector.trim_context(elements, max_tokens=250)

        # Should keep high and medium priority
        self.assertEqual(len(trimmed), 2)
        self.assertEqual(trimmed[0].source, "high")
        self.assertEqual(trimmed[1].source, "medium")

    def test_format_context_for_injection(self):
        """Test formatting context for injection."""
        injector = ContextInjector()

        elements = [
            ContextElement(
                content="Git context", source="git", priority=80, token_estimate=10
            ),
            ContextElement(
                content="File tree", source="tree", priority=70, token_estimate=10
            ),
        ]

        formatted = injector.format_context_for_injection(elements, "Test prompt")

        # Should contain context and prompt
        self.assertIn("Context Injection", formatted)
        self.assertIn("Git context", formatted)
        self.assertIn("File tree", formatted)
        self.assertIn("Test prompt", formatted)

    def test_clear_cache(self):
        """Test clearing cache."""
        injector = ContextInjector()

        # Add to cache
        injector.cache["test"] = ContextElement(
            content="Test", source="test", priority=50, token_estimate=10
        )

        injector.clear_cache()
        self.assertEqual(len(injector.cache), 0)


if __name__ == "__main__":
    unittest.main()
