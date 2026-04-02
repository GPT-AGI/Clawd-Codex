"""Tests for git_context module."""

import unittest
from pathlib import Path

from src.git_context import (
    GitStatus,
    build_git_status,
    get_changed_files,
    get_commit_hash,
    get_current_branch,
    is_git_repo,
    run_git_command,
)


class TestGitContext(unittest.TestCase):
    """Test cases for git context functions."""

    def test_is_git_repo(self):
        """Test git repository detection."""
        # Current project should be a git repo
        self.assertTrue(is_git_repo())

    def test_is_git_repo_nonexistent(self):
        """Test git repository detection for nonexistent path."""
        # Non-existent path should not be a git repo
        self.assertFalse(is_git_repo(Path("/nonexistent/path")))

    def test_get_current_branch(self):
        """Test getting current branch."""
        branch = get_current_branch()
        # Should return a string if we're in a git repo
        self.assertIsNotNone(branch)

    def test_get_commit_hash(self):
        """Test getting commit hash."""
        commit = get_commit_hash()
        # Should return a 40-character SHA if we're in a git repo
        self.assertIsNotNone(commit)
        self.assertEqual(len(commit), 40)

    def test_get_changed_files(self):
        """Test getting changed files."""
        staged, unstaged, untracked = get_changed_files()
        # All should be lists
        self.assertIsInstance(staged, list)
        self.assertIsInstance(unstaged, list)
        self.assertIsInstance(untracked, list)

    def test_build_git_status(self):
        """Test building complete git status."""
        status = build_git_status()

        # Should be a GitStatus instance
        self.assertIsInstance(status, GitStatus)

        # Current project is a git repo
        self.assertTrue(status.is_repo)

        # Should have branch and commit
        self.assertIsNotNone(status.branch)
        self.assertIsNotNone(status.commit_hash)

        # Status should be renderable
        context_str = status.to_context_string()
        self.assertIn("Git Branch:", context_str)
        self.assertIn("Commit:", context_str)

    def test_git_status_non_repo(self):
        """Test GitStatus for non-git directory."""
        status = build_git_status(Path("/tmp"))
        self.assertFalse(status.is_repo)

        # Should indicate not a git repo
        context_str = status.to_context_string()
        self.assertIn("Not a git repository", context_str)

    def test_run_git_command_invalid(self):
        """Test running invalid git command."""
        result = run_git_command(["invalid-command-that-does-not-exist"])
        # Should return None for failed command
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
