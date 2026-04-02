"""
Git repository context collection module.

Provides functions to gather git repository information including:
- Current branch and commit
- Remote repository URL
- Working directory status (clean/dirty)
- Changed files (staged, unstaged, untracked)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GitStatus:
    """Git repository status information."""

    is_repo: bool
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    remote_url: Optional[str] = None
    is_dirty: bool = False
    staged_files: tuple[str, ...] = ()
    unstaged_files: tuple[str, ...] = ()
    untracked_files: tuple[str, ...] = ()

    def to_context_string(self) -> str:
        """Render git status as human-readable context string."""
        if not self.is_repo:
            return "Not a git repository"

        lines = [f"Git Branch: {self.branch or 'unknown'}"]

        if self.commit_hash:
            lines.append(f"Commit: {self.commit_hash[:8]}")

        if self.remote_url:
            lines.append(f"Remote: {self.remote_url}")

        status = "dirty" if self.is_dirty else "clean"
        lines.append(f"Status: {status}")

        if self.staged_files:
            lines.append(f"Staged Files ({len(self.staged_files)}):")
            for file in self.staged_files[:10]:  # Limit to first 10
                lines.append(f"  - {file}")
            if len(self.staged_files) > 10:
                lines.append(f"  ... and {len(self.staged_files) - 10} more")

        if self.unstaged_files:
            lines.append(f"Modified Files ({len(self.unstaged_files)}):")
            for file in self.unstaged_files[:10]:
                lines.append(f"  - {file}")
            if len(self.unstaged_files) > 10:
                lines.append(f"  ... and {len(self.unstaged_files) - 10} more")

        if self.untracked_files:
            lines.append(f"Untracked Files ({len(self.untracked_files)}):")
            for file in self.untracked_files[:10]:
                lines.append(f"  - {file}")
            if len(self.untracked_files) > 10:
                lines.append(f"  ... and {len(self.untracked_files) - 10} more")

        return "\n".join(lines)


def run_git_command(args: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    """
    Run a git command and return its output.

    Args:
        args: Git command arguments (e.g., ['branch', '--show-current'])
        cwd: Working directory (defaults to current directory)

    Returns:
        Command output as string, or None if command failed
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def is_git_repo(path: Optional[Path] = None) -> bool:
    """
    Check if path is inside a git repository.

    Args:
        path: Directory path to check (defaults to current directory)

    Returns:
        True if inside a git repository
    """
    result = run_git_command(["rev-parse", "--is-inside-work-tree"], cwd=path)
    return result == "true"


def get_current_branch(cwd: Optional[Path] = None) -> Optional[str]:
    """Get current git branch name."""
    return run_git_command(["branch", "--show-current"], cwd=cwd)


def get_commit_hash(cwd: Optional[Path] = None) -> Optional[str]:
    """Get current commit hash (full)."""
    return run_git_command(["rev-parse", "HEAD"], cwd=cwd)


def get_remote_url(cwd: Optional[Path] = None) -> Optional[str]:
    """Get remote origin URL."""
    return run_git_command(["config", "--get", "remote.origin.url"], cwd=cwd)


def get_changed_files(cwd: Optional[Path] = None) -> tuple[list[str], list[str], list[str]]:
    """
    Get lists of staged, unstaged, and untracked files.

    Returns:
        Tuple of (staged_files, unstaged_files, untracked_files)
    """
    staged = []
    unstaged = []
    untracked = []

    # Get staged files (index vs HEAD)
    staged_output = run_git_command(["diff", "--name-only", "--cached"], cwd=cwd)
    if staged_output:
        staged = [f for f in staged_output.split("\n") if f]

    # Get unstaged modified files (working tree vs index)
    unstaged_output = run_git_command(["diff", "--name-only"], cwd=cwd)
    if unstaged_output:
        unstaged = [f for f in unstaged_output.split("\n") if f]

    # Get untracked files
    untracked_output = run_git_command(
        ["ls-files", "--others", "--exclude-standard"], cwd=cwd
    )
    if untracked_output:
        untracked = [f for f in untracked_output.split("\n") if f]

    return staged, unstaged, untracked


def build_git_status(repo_path: Optional[Path] = None) -> GitStatus:
    """
    Build complete git status for a repository.

    Args:
        repo_path: Path to repository (defaults to current directory)

    Returns:
        GitStatus object with repository information
    """
    # Check if it's a git repo
    if not is_git_repo(repo_path):
        return GitStatus(is_repo=False)

    # Gather git information
    branch = get_current_branch(repo_path)
    commit_hash = get_commit_hash(repo_path)
    remote_url = get_remote_url(repo_path)

    # Get changed files
    staged, unstaged, untracked = get_changed_files(repo_path)

    # Determine if dirty
    is_dirty = bool(staged or unstaged or untracked)

    return GitStatus(
        is_repo=True,
        branch=branch,
        commit_hash=commit_hash,
        remote_url=remote_url,
        is_dirty=is_dirty,
        staged_files=tuple(staged),
        unstaged_files=tuple(unstaged),
        untracked_files=tuple(untracked),
    )
