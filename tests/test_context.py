"""Tests for extended context module."""

import unittest
from pathlib import Path

from src.context import PortContext, build_port_context, render_context
from src.git_context import GitStatus
from src.file_tree import DirectoryInfo


class TestContext(unittest.TestCase):
    """Test cases for extended context functions."""

    def test_build_port_context_basic(self):
        """Test basic context building."""
        context = build_port_context(include_git=False, include_tree=False)

        # Should be a PortContext instance
        self.assertIsInstance(context, PortContext)

        # Should have basic paths
        self.assertTrue(context.source_root.exists())
        self.assertTrue(context.tests_root.exists())

        # Should have file counts
        self.assertGreater(context.python_file_count, 0)
        self.assertGreater(context.test_file_count, 0)

    def test_build_port_context_with_git(self):
        """Test context building with git status."""
        context = build_port_context(include_git=True, include_tree=False)

        # Should have git status
        self.assertIsNotNone(context.git_status)
        self.assertIsInstance(context.git_status, GitStatus)

        # Current project is a git repo
        self.assertTrue(context.git_status.is_repo)

    def test_build_port_context_with_tree(self):
        """Test context building with file tree."""
        context = build_port_context(include_git=False, include_tree=True)

        # Should have file tree
        self.assertIsNotNone(context.file_tree)
        self.assertIsInstance(context.file_tree, DirectoryInfo)

        # Should have files
        self.assertGreater(context.file_tree.count_files(), 0)

    def test_build_port_context_full(self):
        """Test full context building."""
        context = build_port_context(include_git=True, include_tree=True)

        # Should have both git and tree
        self.assertIsNotNone(context.git_status)
        self.assertIsNotNone(context.file_tree)

    def test_render_context_basic(self):
        """Test basic context rendering."""
        context = build_port_context(include_git=False, include_tree=False)
        rendered = render_context(context, include_tree=False)

        # Should contain basic info
        self.assertIn("Source root:", rendered)
        self.assertIn("Test root:", rendered)
        self.assertIn("Python files:", rendered)

    def test_render_context_with_git(self):
        """Test context rendering with git status."""
        context = build_port_context(include_git=True, include_tree=False)
        rendered = render_context(context, include_tree=False)

        # Should contain git info
        self.assertIn("Git Branch:", rendered)
        self.assertIn("Commit:", rendered)

    def test_render_context_with_tree(self):
        """Test context rendering with file tree."""
        context = build_port_context(include_git=False, include_tree=True)
        rendered = render_context(context, include_tree=True)

        # Should contain file tree info
        self.assertIn("File Tree Summary:", rendered)
        self.assertIn("Total Files:", rendered)
        self.assertIn("Total Directories:", rendered)

    def test_render_context_full(self):
        """Test full context rendering."""
        context = build_port_context(include_git=True, include_tree=True)
        rendered = render_context(context, include_tree=True)

        # Should contain all info
        self.assertIn("Source root:", rendered)
        self.assertIn("Git Branch:", rendered)
        self.assertIn("File Tree Summary:", rendered)

    def test_render_context_tree_limit(self):
        """Test that include_tree parameter controls tree display."""
        context = build_port_context(include_git=False, include_tree=True)

        # Without tree
        rendered_no_tree = render_context(context, include_tree=False)
        self.assertIn("File Tree Summary:", rendered_no_tree)
        # Should only have summary, not full tree

        # With tree
        rendered_with_tree = render_context(context, include_tree=True)
        self.assertIn("File Tree Summary:", rendered_with_tree)
        # Should have more content with full tree

        # With tree should be longer
        self.assertGreater(len(rendered_with_tree), len(rendered_no_tree))

    def test_tree_max_depth_parameter(self):
        """Test tree max depth parameter."""
        context_shallow = build_port_context(
            include_git=False, include_tree=True, tree_max_depth=1
        )
        context_deep = build_port_context(
            include_git=False, include_tree=True, tree_max_depth=3
        )

        # Both should have file trees
        self.assertIsNotNone(context_shallow.file_tree)
        self.assertIsNotNone(context_deep.file_tree)

        # Deeper tree should have more files (or same, but not less)
        # Note: This might not always be true depending on directory structure
        # but it's a reasonable expectation
        shallow_count = context_shallow.file_tree.count_files()
        deep_count = context_deep.file_tree.count_files()
        self.assertGreaterEqual(deep_count, shallow_count)

    def test_context_immutability(self):
        """Test that context is immutable (frozen dataclass)."""
        context = build_port_context()

        # Attempting to modify should raise error
        with self.assertRaises(Exception):  # FrozenInstanceError
            context.python_file_count = 0


if __name__ == "__main__":
    unittest.main()
