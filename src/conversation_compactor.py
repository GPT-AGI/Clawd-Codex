"""
Conversation compaction module for /compact command.

Provides functionality to:
- Compress conversations by token limits
- Preserve key context and recent messages
- Support manual and automatic compaction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent.conversation import Conversation, Message


@dataclass(frozen=True)
class CompactionResult:
    """Result of conversation compaction."""

    original_message_count: int
    compacted_message_count: int
    messages_removed: int
    tokens_saved: int
    strategy: str


class ConversationCompactor:
    """Manages conversation compaction."""

    def __init__(self, max_tokens: int = 100000):
        """
        Initialize compactor.

        Args:
            max_tokens: Maximum tokens before compaction recommended
        """
        self.max_tokens = max_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (4 chars per token)."""
        return len(text) // 4

    def estimate_conversation_tokens(self, conversation: Conversation) -> int:
        """
        Estimate total tokens in conversation.

        Args:
            conversation: Conversation to estimate

        Returns:
            Estimated token count
        """
        total = 0
        for message in conversation.messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total += self.estimate_tokens(block["text"])

        return total

    def should_compact(self, conversation: Conversation) -> bool:
        """
        Check if conversation should be compacted.

        Args:
            conversation: Conversation to check

        Returns:
            True if compaction recommended
        """
        tokens = self.estimate_conversation_tokens(conversation)
        return tokens > self.max_tokens

    def compact_by_tokens(
        self,
        conversation: Conversation,
        max_tokens: int | None = None,
        keep_recent: int = 5,
    ) -> CompactionResult:
        """
        Compact conversation to fit token limit.

        Args:
            conversation: Conversation to compact
            max_tokens: Maximum tokens (defaults to self.max_tokens)
            keep_recent: Number of recent messages to preserve

        Returns:
            CompactionResult
        """
        max_tokens = max_tokens or self.max_tokens
        original_count = len(conversation.messages)

        # Estimate current tokens
        current_tokens = self.estimate_conversation_tokens(conversation)

        if current_tokens <= max_tokens:
            return CompactionResult(
                original_message_count=original_count,
                compacted_message_count=original_count,
                messages_removed=0,
                tokens_saved=0,
                strategy="none_needed",
            )

        # Keep recent messages
        if len(conversation.messages) <= keep_recent:
            return CompactionResult(
                original_message_count=original_count,
                compacted_message_count=original_count,
                messages_removed=0,
                tokens_saved=0,
                strategy="keep_all",
            )

        # Remove older messages
        messages_to_remove = original_count - keep_recent
        conversation.messages = conversation.messages[-keep_recent:]

        # Estimate new token count
        new_tokens = self.estimate_conversation_tokens(conversation)

        return CompactionResult(
            original_message_count=original_count,
            compacted_message_count=len(conversation.messages),
            messages_removed=messages_to_remove,
            tokens_saved=current_tokens - new_tokens,
            strategy="keep_recent",
        )

    def compact_by_time(
        self,
        conversation: Conversation,
        keep_hours: int = 24,
        min_messages: int = 5,
    ) -> CompactionResult:
        """
        Compact conversation by time.

        Args:
            conversation: Conversation to compact
            keep_hours: Keep messages from last N hours
            min_messages: Minimum messages to keep

        Returns:
            CompactionResult
        """
        from datetime import datetime, timedelta

        original_count = len(conversation.messages)

        if original_count <= min_messages:
            return CompactionResult(
                original_message_count=original_count,
                compacted_message_count=original_count,
                messages_removed=0,
                tokens_saved=0,
                strategy="keep_all",
            )

        # This is a simplified implementation
        # In a full implementation, messages would have timestamps
        # For now, keep last min_messages
        conversation.messages = conversation.messages[-min_messages:]

        return CompactionResult(
            original_message_count=original_count,
            compacted_message_count=len(conversation.messages),
            messages_removed=original_count - min_messages,
            tokens_saved=0,  # Would calculate based on actual tokens
            strategy="time_based",
        )

    def compact_by_relevance(
        self,
        conversation: Conversation,
        prompt: str,
        min_messages: int = 5,
    ) -> CompactionResult:
        """
        Compact conversation by relevance to current prompt.

        Args:
            conversation: Conversation to compact
            prompt: Current user prompt
            min_messages: Minimum messages to keep

        Returns:
            CompactionResult
        """
        # Simplified implementation: keep recent messages
        # Full implementation would use semantic similarity
        original_count = len(conversation.messages)

        if original_count <= min_messages:
            return CompactionResult(
                original_message_count=original_count,
                compacted_message_count=original_count,
                messages_removed=0,
                tokens_saved=0,
                strategy="keep_all",
            )

        conversation.messages = conversation.messages[-min_messages:]

        return CompactionResult(
            original_message_count=original_count,
            compacted_message_count=len(conversation.messages),
            messages_removed=original_count - min_messages,
            tokens_saved=0,
            strategy="relevance_based",
        )


def create_compactor(max_tokens: int = 100000) -> ConversationCompactor:
    """
    Create a conversation compactor.

    Args:
        max_tokens: Maximum tokens before compaction

    Returns:
        ConversationCompactor instance
    """
    return ConversationCompactor(max_tokens=max_tokens)
