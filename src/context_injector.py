"""
Context injection module for intelligent context building.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContextElement:
    """A single context element with priority."""

    content: str
    source: str
    priority: int
    token_estimate: int


class ContextInjector:
    """Builds and manages context injection."""

    def __init__(self, max_context_tokens: int = 4000):
        self.max_context_tokens = max_context_tokens
        self.cache = {}

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (4 chars per token)."""
        return len(text) // 4

    def build_git_context(self, git_status) -> ContextElement | None:
        """Build context from git status."""
        if git_status is None or not git_status.is_repo:
            return None

        lines = ["## Git Repository Status", git_status.to_context_string()]
        content = "\n".join(lines)

        return ContextElement(
            content=content,
            source="git_status",
            priority=80,
            token_estimate=self.estimate_tokens(content),
        )

    def build_file_tree_context(self, file_tree, include_files: bool = False) -> ContextElement | None:
        """Build context from file tree."""
        if file_tree is None:
            return None

        lines = [
            "## Project Structure",
            f"Files: {file_tree.count_files()}, Directories: {file_tree.count_dirs()}",
        ]

        if include_files:
            lines.append("")
            lines.append(file_tree.to_tree_string(show_files=True))

        content = "\n".join(lines)
        return ContextElement(
            content=content,
            source="file_tree",
            priority=70,
            token_estimate=self.estimate_tokens(content),
        )

    def build_readme_context(self, workspace_root: Path, max_length: int = 2000) -> ContextElement | None:
        """Build context from README file."""
        readme_paths = [workspace_root / "README.md", workspace_root / "readme.md"]

        for readme_path in readme_paths:
            if readme_path.exists():
                try:
                    content = readme_path.read_text()[:max_length]
                    lines = ["## Project Documentation", f"Source: {readme_path.name}", "", content]
                    full_content = "\n".join(lines)

                    return ContextElement(
                        content=full_content,
                        source="readme",
                        priority=60,
                        token_estimate=self.estimate_tokens(full_content),
                    )
                except OSError:
                    continue

        return None

    def build_claudemd_context(self, workspace_root: Path, max_length: int = 3000) -> ContextElement | None:
        """Build context from CLAUDE.md file."""
        claude_md = workspace_root / "CLAUDE.md"
        if not claude_md.exists():
            return None

        try:
            content = claude_md.read_text()[:max_length]
            lines = ["## Project Instructions (CLAUDE.md)", "", content]
            full_content = "\n".join(lines)

            return ContextElement(
                content=full_content,
                source="claude_md",
                priority=90,  # Highest priority
                token_estimate=self.estimate_tokens(full_content),
            )
        except OSError:
            return None

    def build_workspace_context(
        self,
        workspace_context,
        include_git: bool = True,
        include_tree: bool = True,
        include_readme: bool = True,
        include_claude_md: bool = True,
    ) -> list[ContextElement]:
        """Build all context elements from workspace."""
        elements = []
        workspace_root = workspace_context.source_root.parent

        # Add CLAUDE.md (highest priority)
        if include_claude_md:
            claude_md = self.build_claudemd_context(workspace_root)
            if claude_md:
                elements.append(claude_md)

        # Add git status
        if include_git:
            git_context = self.build_git_context(workspace_context.git_status)
            if git_context:
                elements.append(git_context)

        # Add file tree
        if include_tree:
            tree_context = self.build_file_tree_context(workspace_context.file_tree, include_files=False)
            if tree_context:
                elements.append(tree_context)

        # Add README
        if include_readme:
            readme_context = self.build_readme_context(workspace_root)
            if readme_context:
                elements.append(readme_context)

        # Sort by priority (descending)
        elements.sort(key=lambda e: e.priority, reverse=True)

        return elements

    def trim_context(self, elements: list[ContextElement], max_tokens: int | None = None) -> list[ContextElement]:
        """Trim context elements to fit token limit (assumes already sorted by priority)."""
        max_tokens = max_tokens or self.max_context_tokens

        trimmed = []
        total_tokens = 0

        # Assume elements are already sorted by priority (descending)
        for element in elements:
            if total_tokens + element.token_estimate <= max_tokens:
                trimmed.append(element)
                total_tokens += element.token_estimate

        return trimmed

    def format_context_for_injection(self, elements: list[ContextElement], prompt: str) -> str:
        """Format context elements for injection into prompt."""
        if not elements:
            return prompt

        context_parts = [element.content for element in elements]
        context_content = "\n\n".join(context_parts)

        return f"""## Context Injection

{context_content}

## Your Task

{prompt}"""

    def clear_cache(self) -> None:
        """Clear context cache."""
        self.cache.clear()
