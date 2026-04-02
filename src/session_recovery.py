"""
Session recovery module for /resume command.

Provides functionality to:
- List available sessions with metadata
- Display session summaries
- Resume sessions with full context restoration
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent.session import Session


@dataclass(frozen=True)
class ResumableSession:
    """A session available for resumption."""

    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    provider: str
    model: str
    preview: str
    file_path: Path

    def to_display_string(self, index: int | None = None) -> str:
        """Format session for display."""
        prefix = f"{index}. " if index is not None else ""
        lines = [
            f"{prefix}Session: {self.session_id[:8]}",
            f"   Created: {self._format_timestamp(self.created_at)}",
            f"   Updated: {self._format_timestamp(self.updated_at)}",
            f"   Provider: {self.provider} / {self.model}",
            f"   Messages: {self.message_count}",
            f"   Preview: {self.preview[:100]}{'...' if len(self.preview) > 100 else ''}",
        ]
        return "\n".join(lines)

    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            return timestamp


class SessionRecovery:
    """Manages session recovery operations."""

    def __init__(self, sessions_dir: Path):
        """
        Initialize session recovery manager.

        Args:
            sessions_dir: Directory containing session files
        """
        self.sessions_dir = sessions_dir

    def list_resumable_sessions(
        self, limit: int = 10, sort_by: str = "updated_at"
    ) -> list[ResumableSession]:
        """
        List sessions available for resumption.

        Args:
            limit: Maximum number of sessions to return
            sort_by: Field to sort by (updated_at, created_at, message_count)

        Returns:
            List of ResumableSession objects
        """
        sessions = []

        # Find all session files
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                session = self._load_session_metadata(session_file)
                if session:
                    sessions.append(session)
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort sessions
        sessions.sort(
            key=lambda s: getattr(s, sort_by, s.updated_at), reverse=True
        )

        # Apply limit
        return sessions[:limit]

    def get_session(self, session_id: str) -> ResumableSession | None:
        """
        Get a specific session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ResumableSession or None if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            # Try partial match
            matches = list(self.sessions_dir.glob(f"{session_id}*.json"))
            if matches:
                session_file = matches[0]
            else:
                return None

        return self._load_session_metadata(session_file)

    def _load_session_metadata(self, session_file: Path) -> ResumableSession | None:
        """
        Load session metadata from file.

        Args:
            session_file: Path to session JSON file

        Returns:
            ResumableSession or None if invalid
        """
        try:
            with open(session_file) as f:
                data = json.load(f)

            # Extract metadata
            session_id = data.get("session_id", session_file.stem)
            messages = data.get("messages", [])

            # Generate preview
            preview = self._generate_preview(messages)

            return ResumableSession(
                session_id=session_id,
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
                message_count=len(messages),
                provider=data.get("provider", "unknown"),
                model=data.get("model", "unknown"),
                preview=preview,
                file_path=session_file,
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _generate_preview(self, messages: list, max_length: int = 150) -> str:
        """Generate preview from messages."""
        if not messages:
            return "Empty session"

        # Find first user message
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content[:max_length] + (
                        "..." if len(content) > max_length else ""
                    )
            elif isinstance(message, str):
                return message[:max_length] + (
                    "..." if len(message) > max_length else ""
                )

        return "No preview available"

    def display_sessions(self, sessions: list[ResumableSession]) -> str:
        """
        Format sessions for display.

        Args:
            sessions: List of sessions to display

        Returns:
            Formatted string
        """
        if not sessions:
            return "No resumable sessions found."

        lines = ["Available Sessions:", ""]
        for i, session in enumerate(sessions, 1):
            lines.append(session.to_display_string(index=i))
            lines.append("")

        lines.append("Use '/resume <session-id>' to resume a session.")
        return "\n".join(lines)


def create_recovery_manager(sessions_dir: Path | None = None) -> SessionRecovery:
    """
    Create a session recovery manager.

    Args:
        sessions_dir: Optional sessions directory (defaults to .port_sessions)

    Returns:
        SessionRecovery instance
    """
    if sessions_dir is None:
        sessions_dir = Path(".port_sessions")

    return SessionRecovery(sessions_dir)
