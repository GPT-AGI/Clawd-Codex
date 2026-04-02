"""Tests for session_index module."""

import unittest
from pathlib import Path
from datetime import datetime

from src.session_index import SessionIndex, SessionSummary


class TestSessionIndex(unittest.TestCase):
    """Test cases for session indexing."""

    def test_create_session_summary(self):
        """Test creating a session summary."""
        summary = SessionSummary(
            session_id="test-123",
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T01:00:00",
            provider="anthropic",
            model="claude-sonnet-4-6",
            message_count=10,
            total_tokens=5000,
            total_cost_usd=0.05,
            tags=("test", "demo"),
            preview="Test session preview",
        )

        self.assertEqual(summary.session_id, "test-123")
        self.assertEqual(summary.provider, "anthropic")
        self.assertEqual(summary.tags, ("test", "demo"))

    def test_session_summary_to_dict(self):
        """Test session summary serialization."""
        summary = SessionSummary(
            session_id="test-123",
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T01:00:00",
            provider="anthropic",
            model="claude-sonnet-4-6",
            message_count=10,
            total_tokens=5000,
            total_cost_usd=0.05,
            tags=("test",),
            preview="Test",
        )

        data = summary.to_dict()
        self.assertEqual(data["session_id"], "test-123")
        self.assertEqual(data["tags"], ["test"])

    def test_session_summary_from_dict(self):
        """Test session summary deserialization."""
        data = {
            "session_id": "test-456",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T01:00:00",
            "provider": "openai",
            "model": "gpt-4o",
            "message_count": 5,
            "total_tokens": 2000,
            "total_cost_usd": 0.03,
            "tags": ["prod"],
            "preview": "Production session",
        }

        summary = SessionSummary.from_dict(data)
        self.assertEqual(summary.session_id, "test-456")
        self.assertEqual(summary.tags, ("prod",))

    def test_create_session_index(self):
        """Test creating a session index."""
        index = SessionIndex()
        self.assertEqual(len(index.sessions), 0)

    def test_add_session(self):
        """Test adding a session to index."""
        index = SessionIndex()
        summary = SessionSummary(
            session_id="test-123",
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T01:00:00",
            provider="anthropic",
            model="claude-sonnet-4-6",
            message_count=10,
            total_tokens=5000,
            total_cost_usd=0.05,
            tags=(),
            preview="Test",
        )

        index.add_session(summary)
        self.assertEqual(len(index.sessions), 1)
        self.assertIn("test-123", index.sessions)

    def test_remove_session(self):
        """Test removing a session from index."""
        index = SessionIndex()
        summary = SessionSummary(
            session_id="test-123",
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T01:00:00",
            provider="anthropic",
            model="claude-sonnet-4-6",
            message_count=10,
            total_tokens=5000,
            total_cost_usd=0.05,
            tags=(),
            preview="Test",
        )

        index.add_session(summary)
        index.remove_session("test-123")
        self.assertEqual(len(index.sessions), 0)

    def test_get_session(self):
        """Test getting a session by ID."""
        index = SessionIndex()
        summary = SessionSummary(
            session_id="test-123",
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T01:00:00",
            provider="anthropic",
            model="claude-sonnet-4-6",
            message_count=10,
            total_tokens=5000,
            total_cost_usd=0.05,
            tags=(),
            preview="Test",
        )

        index.add_session(summary)
        retrieved = index.get_session("test-123")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.session_id, "test-123")

    def test_list_sessions(self):
        """Test listing sessions with sorting."""
        index = SessionIndex()

        # Add multiple sessions
        for i in range(5):
            summary = SessionSummary(
                session_id=f"test-{i}",
                created_at=f"2025-01-0{i}T00:00:00",
                updated_at=f"2025-01-0{i}T01:00:00",
                provider="anthropic",
                model="claude-sonnet-4-6",
                message_count=i * 10,
                total_tokens=i * 1000,
                total_cost_usd=i * 0.01,
                tags=(),
                preview=f"Test {i}",
            )
            index.add_session(summary)

        # List all sessions
        sessions = index.list_sessions()
        self.assertEqual(len(sessions), 5)

        # List with limit
        sessions = index.list_sessions(limit=3)
        self.assertEqual(len(sessions), 3)

    def test_list_sessions_sorting(self):
        """Test session list sorting."""
        index = SessionIndex()

        # Add sessions with different costs
        for i, cost in enumerate([0.05, 0.15, 0.10]):
            summary = SessionSummary(
                session_id=f"test-{i}",
                created_at="2025-01-01T00:00:00",
                updated_at="2025-01-01T01:00:00",
                provider="anthropic",
                model="claude-sonnet-4-6",
                message_count=10,
                total_tokens=1000,
                total_cost_usd=cost,
                tags=(),
                preview=f"Test {i}",
            )
            index.add_session(summary)

        # Sort by cost (ascending)
        sessions = index.list_sessions(sort_by="total_cost_usd", reverse=False)
        self.assertEqual(sessions[0].total_cost_usd, 0.05)
        self.assertEqual(sessions[1].total_cost_usd, 0.10)
        self.assertEqual(sessions[2].total_cost_usd, 0.15)

    def test_get_statistics(self):
        """Test getting aggregate statistics."""
        index = SessionIndex()

        # Add sessions
        for i in range(3):
            summary = SessionSummary(
                session_id=f"test-{i}",
                created_at="2025-01-01T00:00:00",
                updated_at="2025-01-01T01:00:00",
                provider="anthropic",
                model="claude-sonnet-4-6",
                message_count=10,
                total_tokens=1000,
                total_cost_usd=0.05,
                tags=(),
                preview=f"Test {i}",
            )
            index.add_session(summary)

        stats = index.get_statistics()
        self.assertEqual(stats["total_sessions"], 3)
        self.assertEqual(stats["total_messages"], 30)
        self.assertEqual(stats["total_tokens"], 3000)
        self.assertAlmostEqual(stats["total_cost_usd"], 0.15, places=2)


if __name__ == "__main__":
    unittest.main()
