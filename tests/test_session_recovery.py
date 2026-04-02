"""Tests for session_recovery module."""

import unittest
from pathlib import Path
from datetime import datetime

from src.session_recovery import (
    ResumableSession,
    SessionRecovery,
    create_recovery_manager,
)


class TestSessionRecovery(unittest.TestCase):
    """Test cases for session recovery."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / "test_sessions"
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test files."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_create_resumable_session(self):
        """Test creating a resumable session."""
        session = ResumableSession(
            session_id="test-123",
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T01:00:00",
            message_count=10,
            provider="anthropic",
            model="claude-sonnet-4-6",
            preview="Test session preview",
            file_path=Path("test.json"),
        )

        self.assertEqual(session.session_id, "test-123")
        self.assertEqual(session.message_count, 10)

    def test_resumable_session_display_string(self):
        """Test session display formatting."""
        session = ResumableSession(
            session_id="test-456",
            created_at="2025-01-01T10:30:00",
            updated_at="2025-01-01T11:45:00",
            message_count=5,
            provider="openai",
            model="gpt-4o",
            preview="This is a test session with a longer preview that should be truncated",
            file_path=Path("test.json"),
        )

        display = session.to_display_string(index=1)

        self.assertIn("test-456", display)
        self.assertIn("2025-01-01", display)
        self.assertIn("openai", display)
        self.assertIn("gpt-4o", display)

    def test_create_recovery_manager(self):
        """Test creating recovery manager."""
        recovery = create_recovery_manager(self.test_dir)
        self.assertIsInstance(recovery, SessionRecovery)
        self.assertEqual(recovery.sessions_dir, self.test_dir)

    def test_list_resumable_sessions_empty(self):
        """Test listing sessions when none exist."""
        recovery = SessionRecovery(self.test_dir)
        sessions = recovery.list_resumable_sessions()
        self.assertEqual(len(sessions), 0)

    def test_list_resumable_sessions_with_data(self):
        """Test listing sessions with actual data."""
        # Create test session file
        import json

        session_data = {
            "session_id": "test-789",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T01:00:00",
            "messages": [
                {"role": "user", "content": "Test message"},
                {"role": "assistant", "content": "Response"},
            ],
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        }

        session_file = self.test_dir / "test-789.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f)

        recovery = SessionRecovery(self.test_dir)
        sessions = recovery.list_resumable_sessions()

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].session_id, "test-789")
        self.assertEqual(sessions[0].message_count, 2)

    def test_get_session_by_id(self):
        """Test getting a specific session."""
        import json

        session_data = {
            "session_id": "test-specific",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T01:00:00",
            "messages": [{"role": "user", "content": "Hello"}],
            "provider": "openai",
            "model": "gpt-4o",
        }

        session_file = self.test_dir / "test-specific.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f)

        recovery = SessionRecovery(self.test_dir)
        session = recovery.get_session("test-specific")

        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, "test-specific")

    def test_get_session_partial_id(self):
        """Test getting session with partial ID."""
        import json

        session_data = {
            "session_id": "abc123def456",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T01:00:00",
            "messages": [],
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        }

        session_file = self.test_dir / "abc123def456.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f)

        recovery = SessionRecovery(self.test_dir)

        # Test partial match
        session = recovery.get_session("abc123")
        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, "abc123def456")

    def test_display_sessions(self):
        """Test formatting sessions for display."""
        sessions = [
            ResumableSession(
                session_id="test-1",
                created_at="2025-01-01T00:00:00",
                updated_at="2025-01-01T01:00:00",
                message_count=5,
                provider="anthropic",
                model="claude-sonnet-4-6",
                preview="Test 1",
                file_path=Path("test1.json"),
            ),
            ResumableSession(
                session_id="test-2",
                created_at="2025-01-02T00:00:00",
                updated_at="2025-01-02T01:00:00",
                message_count=10,
                provider="openai",
                model="gpt-4o",
                preview="Test 2",
                file_path=Path("test2.json"),
            ),
        ]

        recovery = SessionRecovery(self.test_dir)
        display = recovery.display_sessions(sessions)

        self.assertIn("Available Sessions:", display)
        self.assertIn("test-1", display)
        self.assertIn("test-2", display)
        self.assertIn("/resume", display)

    def test_display_sessions_empty(self):
        """Test displaying empty session list."""
        recovery = SessionRecovery(self.test_dir)
        display = recovery.display_sessions([])

        self.assertIn("No resumable sessions", display)


if __name__ == "__main__":
    unittest.main()
