"""Tests for file_tree module."""

import unittest
from pathlib import Path

from src.file_tree import (
    DirectoryInfo,
    FileInfo,
    build_file_tree,
    categorize_file,
    render_file_tree,
    should_ignore,
)


class TestFileTree(unittest.TestCase):
    """Test cases for file tree functions."""

    def test_categorize_file_code(self):
        """Test file categorization for code files."""
        self.assertEqual(categorize_file(Path("test.py")), "code")
        self.assertEqual(categorize_file(Path("app.js")), "code")
        self.assertEqual(categorize_file(Path("main.go")), "code")

    def test_categorize_file_config(self):
        """Test file categorization for config files."""
        self.assertEqual(categorize_file(Path("config.json")), "config")
        self.assertEqual(categorize_file(Path("settings.yaml")), "config")
        self.assertEqual(categorize_file(Path("pyproject.toml")), "config")

    def test_categorize_file_docs(self):
        """Test file categorization for documentation files."""
        self.assertEqual(categorize_file(Path("README.md")), "docs")
        self.assertEqual(categorize_file(Path("docs.txt")), "docs")

    def test_categorize_file_tests(self):
        """Test file categorization for test files."""
        self.assertEqual(categorize_file(Path("test_main.py")), "tests")
        self.assertEqual(categorize_file(Path("app_test.js")), "tests")
        self.assertEqual(categorize_file(Path("tests/unit/test_foo.py")), "tests")

    def test_categorize_file_other(self):
        """Test file categorization for other files."""
        self.assertEqual(categorize_file(Path("image.png")), "other")
        self.assertEqual(categorize_file(Path("data.csv")), "other")

    def test_should_ignore(self):
        """Test ignore patterns."""
        self.assertTrue(should_ignore(Path("__pycache__")))
        self.assertTrue(should_ignore(Path(".git")))
        self.assertTrue(should_ignore(Path("node_modules")))
        self.assertTrue(should_ignore(Path(".venv")))

    def test_should_not_ignore(self):
        """Test non-ignored directories."""
        self.assertFalse(should_ignore(Path("src")))
        self.assertFalse(should_ignore(Path("tests")))
        self.assertFalse(should_ignore(Path("myproject")))

    def test_build_file_tree(self):
        """Test building file tree for current project."""
        root = Path(__file__).resolve().parent.parent
        tree = build_file_tree(root, max_depth=2)

        # Should be a DirectoryInfo
        self.assertIsInstance(tree, DirectoryInfo)

        # Should have files and/or subdirectories
        self.assertTrue(tree.files or tree.subdirs)

        # Should be able to count files
        file_count = tree.count_files()
        self.assertGreater(file_count, 0)

    def test_file_info_to_tree_entry(self):
        """Test file info tree entry rendering."""
        file_info = FileInfo(
            path=Path("test.py"),
            relative_path="test.py",
            size_bytes=1024,
            category="code",
        )

        entry = file_info.to_tree_entry()
        self.assertIn("test.py", entry)
        self.assertIn("code", entry)
        self.assertIn("1.0KB", entry)

    def test_directory_info_to_tree_string(self):
        """Test directory tree string rendering."""
        files = (
            FileInfo(
                path=Path("test.py"),
                relative_path="test.py",
                size_bytes=1024,
                category="code",
            ),
        )

        dir_info = DirectoryInfo(
            path=Path("myproject"),
            relative_path=".",
            files=files,
            subdirs=(),
        )

        tree_str = dir_info.to_tree_string()
        self.assertIn("myproject", tree_str)
        self.assertIn("test.py", tree_str)

    def test_render_file_tree(self):
        """Test full file tree rendering."""
        root = Path(__file__).resolve().parent.parent
        tree_str = render_file_tree(root, max_depth=2)

        # Should contain project structure info
        self.assertIn("Project Structure:", tree_str)
        self.assertIn("Total Files:", tree_str)
        self.assertIn("Total Directories:", tree_str)

    def test_max_depth_limit(self):
        """Test that max depth is respected."""
        root = Path(__file__).resolve().parent.parent
        tree = build_file_tree(root, max_depth=1)

        # At depth 1, we should have limited subdirectories
        # All subdirs should have depth 0 relative to their parent
        def check_depth(dir_info: DirectoryInfo, current_depth: int) -> int:
            max_found = current_depth
            for subdir in dir_info.subdirs:
                max_found = max(max_found, check_depth(subdir, current_depth + 1))
            return max_found

        max_depth_found = check_depth(tree, 0)
        self.assertLessEqual(max_depth_found, 1)


if __name__ == "__main__":
    unittest.main()
