"""Tests for conversation_compactor module."""

import unittest
from unittest.mock import MagicMock

from src.conversation_compactor import (
    CompactionResult,
    ConversationCompactor,
    create_compactor,
)


class TestConversationCompactor(unittest.TestCase):
    """Test cases for conversation compaction."""

    def create_mock_conversation(self, message_count: int):
        """Create a mock conversation."""
        conversation = MagicMock()
        conversation.messages = [
            {"role": "user", "content": f"Message {i} " * 100}  # ~500 chars each
            for i in range(message_count)
        ]
        return conversation

    def test_create_compactor(self):
        """Test creating a compactor."""
        compactor = create_compactor(max_tokens=50000)
        self.assertIsInstance(compactor, ConversationCompactor)
        self.assertEqual(compactor.max_tokens, 50000)

    def test_estimate_tokens(self):
        """Test token estimation."""
        compactor = ConversationCompactor()

        # Empty string
        self.assertEqual(compactor.estimate_tokens(""), 0)

        # Known length
        text = "Hello world"  # 11 chars
        estimate = compactor.estimate_tokens(text)
        self.assertEqual(estimate, 2)  # 11 // 4 = 2

    def test_estimate_conversation_tokens(self):
        """Test conversation token estimation."""
        compactor = ConversationCompactor()
        conversation = self.create_mock_conversation(10)

        tokens = compactor.estimate_conversation_tokens(conversation)

        # Each message has ~1200 chars = ~300 tokens
        # 10 messages = ~3000 tokens
        self.assertGreater(tokens, 2000)
        self.assertLess(tokens, 4000)

    def test_should_compact(self):
        """Test compaction recommendation."""
        compactor = ConversationCompactor(max_tokens=5000)

        # Small conversation (5 messages * ~300 tokens = ~1500 tokens)
        small = self.create_mock_conversation(5)
        self.assertFalse(compactor.should_compact(small))

        # Large conversation (100 messages * ~300 tokens = ~30000 tokens)
        large = self.create_mock_conversation(100)
        self.assertTrue(compactor.should_compact(large))

    def test_compact_by_tokens_no_compaction_needed(self):
        """Test compaction when not needed."""
        compactor = ConversationCompactor(max_tokens=10000)
        conversation = self.create_mock_conversation(5)

        result = compactor.compact_by_tokens(conversation)

        self.assertEqual(result.strategy, "none_needed")
        self.assertEqual(result.messages_removed, 0)

    def test_compact_by_tokens_with_compaction(self):
        """Test compaction with token limit."""
        compactor = ConversationCompactor(max_tokens=500)
        conversation = self.create_mock_conversation(20)

        result = compactor.compact_by_tokens(conversation, keep_recent=3)

        self.assertEqual(result.strategy, "keep_recent")
        self.assertGreater(result.messages_removed, 0)
        self.assertEqual(len(conversation.messages), 3)

    def test_compact_by_tokens_preserves_recent(self):
        """Test that compaction preserves recent messages."""
        compactor = ConversationCompactor(max_tokens=500)
        conversation = self.create_mock_conversation(20)

        original_messages = list(conversation.messages)
        compactor.compact_by_tokens(conversation, keep_recent=5)

        # Last 5 messages should be preserved
        for i, msg in enumerate(conversation.messages):
            self.assertEqual(msg, original_messages[-(5 - i)])

    def test_compact_by_time(self):
        """Test time-based compaction."""
        compactor = ConversationCompactor()
        conversation = self.create_mock_conversation(20)

        result = compactor.compact_by_time(conversation, keep_hours=24, min_messages=5)

        self.assertEqual(result.strategy, "time_based")
        self.assertEqual(len(conversation.messages), 5)

    def test_compact_by_relevance(self):
        """Test relevance-based compaction."""
        compactor = ConversationCompactor()
        conversation = self.create_mock_conversation(20)

        result = compactor.compact_by_relevance(
            conversation, prompt="test prompt", min_messages=5
        )

        self.assertEqual(result.strategy, "relevance_based")
        self.assertEqual(len(conversation.messages), 5)

    def test_compaction_result(self):
        """Test compaction result data."""
        result = CompactionResult(
            original_message_count=20,
            compacted_message_count=5,
            messages_removed=15,
            tokens_saved=1500,
            strategy="keep_recent",
        )

        self.assertEqual(result.original_message_count, 20)
        self.assertEqual(result.compacted_message_count, 5)
        self.assertEqual(result.messages_removed, 15)
        self.assertEqual(result.tokens_saved, 1500)
        self.assertEqual(result.strategy, "keep_recent")


if __name__ == "__main__":
    unittest.main()
