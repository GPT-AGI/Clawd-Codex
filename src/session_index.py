"""
Session indexing and search module.

Provides functionality to:
- Index session metadata for fast searching
- Search sessions by various criteria
- Generate session summaries
- Manage session lifecycle
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class SessionSummary:
    """Summary information for a session."""

    session_id: str
    created_at: str
    updated_at: str
    provider: str
    model: str
    message_count: int
    total_tokens: int
    total_cost_usd: float
    tags: tuple[str, ...] = ()
    preview: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provider": self.provider,
            "model": self.model,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "tags": list(self.tags),
            "preview": self.preview,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionSummary:
        """Create from dictionary."""
        data["tags"] = tuple(data.get("tags", []))
        return cls(**data)


@dataclass
class SessionIndex:
    """Index of session metadata."""

    sessions: dict[str, SessionSummary] = field(default_factory=dict)
    index_path: Path | None = None

    def add_session(self, summary: SessionSummary) -> None:
        """Add or update session in index."""
        self.sessions[summary.session_id] = summary

    def remove_session(self, session_id: str) -> None:
        """Remove session from index."""
        self.sessions.pop(session_id, None)

    def get_session(self, session_id: str) -> SessionSummary | None:
        """Get session summary by ID."""
        return self.sessions.get(session_id)

    def list_sessions(
        self,
        limit: int | None = None,
        sort_by: str = "updated_at",
        reverse: bool = True,
    ) -> list[SessionSummary]:
        """
        List sessions with optional sorting and limiting.

        Args:
            limit: Maximum number of sessions to return
            sort_by: Field to sort by (created_at, updated_at, message_count, total_cost_usd)
            reverse: Sort in descending order

        Returns:
            List of SessionSummary objects
        """
        sessions = list(self.sessions.values())

        # Sort sessions
        sessions.sort(
            key=lambda s: getattr(s, sort_by, s.updated_at),
            reverse=reverse,
        )

        # Apply limit
        if limit is not None:
            sessions = sessions[:limit]

        return sessions

    def search_sessions(
        self,
        query: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        min_cost: float | None = None,
        max_cost: float | None = None,
        tags: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[SessionSummary]:
        """
        Search sessions by various criteria.

        Args:
            query: Text to search in session preview
            provider: Filter by provider
            model: Filter by model
            min_tokens: Minimum total tokens
            max_tokens: Maximum total tokens
            min_cost: Minimum cost in USD
            max_cost: Maximum cost in USD
            tags: Filter by tags (session must have all tags)
            start_date: Filter sessions created after this date
            end_date: Filter sessions created before this date

        Returns:
            List of matching SessionSummary objects
        """
        results = []

        for session in self.sessions.values():
            # Apply filters
            if query is not None and query.lower() not in session.preview.lower():
                continue

            if provider is not None and session.provider != provider:
                continue

            if model is not None and session.model != model:
                continue

            if min_tokens is not None and session.total_tokens < min_tokens:
                continue

            if max_tokens is not None and session.total_tokens > max_tokens:
                continue

            if min_cost is not None and session.total_cost_usd < min_cost:
                continue

            if max_cost is not None and session.total_cost_usd > max_cost:
                continue

            if tags is not None and not all(tag in session.tags for tag in tags):
                continue

            if start_date is not None and session.created_at < start_date:
                continue

            if end_date is not None and session.created_at > end_date:
                continue

            results.append(session)

        # Sort by updated_at descending
        results.sort(key=lambda s: s.updated_at, reverse=True)

        return results

    def save(self, path: Path | None = None) -> None:
        """
        Save index to JSON file.

        Args:
            path: Path to save index (defaults to self.index_path)
        """
        save_path = path or self.index_path
        if save_path is None:
            raise ValueError("No index path specified")

        data = {
            "sessions": {sid: summary.to_dict() for sid, summary in self.sessions.items()}
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> SessionIndex:
        """
        Load index from JSON file.

        Args:
            path: Path to load index from

        Returns:
            SessionIndex instance
        """
        if not path.exists():
            return cls(index_path=path)

        with open(path) as f:
            data = json.load(f)

        sessions = {
            sid: SessionSummary.from_dict(summary_data)
            for sid, summary_data in data.get("sessions", {}).items()
        }

        return cls(sessions=sessions, index_path=path)

    def get_statistics(self) -> dict:
        """Get aggregate statistics about sessions."""
        if not self.sessions:
            return {
                "total_sessions": 0,
                "total_messages": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            }

        return {
            "total_sessions": len(self.sessions),
            "total_messages": sum(s.message_count for s in self.sessions.values()),
            "total_tokens": sum(s.total_tokens for s in self.sessions.values()),
            "total_cost_usd": sum(s.total_cost_usd for s in self.sessions.values()),
            "providers": list(set(s.provider for s in self.sessions.values())),
            "models": list(set(s.model for s in self.sessions.values())),
        }


def index_sessions_from_directory(
    sessions_dir: Path,
    index_path: Path | None = None,
) -> SessionIndex:
    """
    Build session index from sessions directory.

    Args:
        sessions_dir: Directory containing session JSON files
        index_path: Optional path to save index

    Returns:
        SessionIndex instance
    """
    index = SessionIndex(index_path=index_path)

    # Find all session files
    for session_file in sessions_dir.glob("*.json"):
        try:
            with open(session_file) as f:
                data = json.load(f)

            # Extract summary information
            summary = SessionSummary(
                session_id=data.get("session_id", session_file.stem),
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
                provider=data.get("provider", "unknown"),
                model=data.get("model", "unknown"),
                message_count=len(data.get("messages", [])),
                total_tokens=data.get("input_tokens", 0) + data.get("output_tokens", 0),
                total_cost_usd=data.get("cost_usd", 0.0),
                tags=tuple(data.get("tags", [])),
                preview=_generate_preview(data.get("messages", [])),
            )

            index.add_session(summary)
        except (json.JSONDecodeError, KeyError):
            # Skip malformed session files
            continue

    return index


def _generate_preview(messages: list[dict], max_length: int = 200) -> str:
    """Generate preview text from messages."""
    if not messages:
        return ""

    # Find first user message
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content[:max_length] + ("..." if len(content) > max_length else "")

    return ""
