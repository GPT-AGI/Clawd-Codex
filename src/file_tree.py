"""
File tree construction module.

Provides functionality to build and render directory tree structures with:
- Configurable depth limits
- File categorization (code, config, docs, tests)
- Size and count limits
- Tree visualization
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class FileInfo:
    """Information about a single file."""

    path: Path
    relative_path: str
    size_bytes: int
    category: str

    def to_tree_entry(self, prefix: str = "", is_last: bool = False) -> str:
        """Render file as tree entry."""
        connector = "└── " if is_last else "├── "
        size_kb = self.size_bytes / 1024
        return f"{prefix}{connector}{self.path.name} ({size_kb:.1f}KB, {self.category})"


@dataclass(frozen=True)
class DirectoryInfo:
    """Information about a directory."""

    path: Path
    relative_path: str
    files: tuple[FileInfo, ...]
    subdirs: tuple[DirectoryInfo, ...]

    def to_tree_string(
        self, prefix: str = "", is_last: bool = False, show_files: bool = True
    ) -> str:
        """Render directory as tree string."""
        lines = []

        # Directory header
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{self.path.name}/")

        # Prepare prefix for children
        if is_last:
            child_prefix = prefix + "    "
        else:
            child_prefix = prefix + "│   "

        if show_files:
            # Render files
            for i, file_info in enumerate(self.files):
                is_last_file = i == len(self.files) - 1 and not self.subdirs
                lines.append(file_info.to_tree_entry(child_prefix, is_last_file))

        # Render subdirectories
        for i, subdir in enumerate(self.subdirs):
            is_last_subdir = i == len(self.subdirs) - 1
            lines.append(
                subdir.to_tree_string(child_prefix, is_last_subdir, show_files)
            )

        return "\n".join(lines)

    def count_files(self) -> int:
        """Count total files in directory and subdirectories."""
        return len(self.files) + sum(subdir.count_files() for subdir in self.subdirs)

    def count_dirs(self) -> int:
        """Count total subdirectories."""
        return len(self.subdirs) + sum(
            subdir.count_dirs() for subdir in self.subdirs
        )


# File category mappings
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}

CONFIG_EXTENSIONS = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
}

DOC_EXTENSIONS = {
    ".md",
    ".rst",
    ".txt",
    ".pdf",
}

TEST_PATTERNS = {
    "test_",
    "_test.",
    "tests/",
    "test/",
    "__tests__/",
}


def categorize_file(path: Path) -> str:
    """
    Categorize file by extension and path.

    Args:
        path: File path

    Returns:
        Category string: 'code', 'config', 'docs', 'tests', or 'other'
    """
    # Check if test file
    path_str = str(path)
    if any(pattern in path_str for pattern in TEST_PATTERNS):
        return "tests"

    # Check by extension
    ext = path.suffix.lower()

    if ext in CODE_EXTENSIONS:
        return "code"
    elif ext in CONFIG_EXTENSIONS:
        return "config"
    elif ext in DOC_EXTENSIONS:
        return "docs"
    else:
        return "other"


def should_ignore(path: Path) -> bool:
    """
    Check if path should be ignored in tree.

    Args:
        path: Path to check

    Returns:
        True if path should be ignored
    """
    # Ignore patterns
    ignore_names = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
        "*.egg-info",
        ".eggs",
    }

    return path.name in ignore_names


def build_file_tree(
    root: Path,
    max_depth: int = 3,
    max_files_per_dir: int = 50,
    max_file_size_kb: int = 1024,
) -> DirectoryInfo:
    """
    Build file tree structure for a directory.

    Args:
        root: Root directory path
        max_depth: Maximum depth to traverse
        max_files_per_dir: Maximum files to include per directory
        max_file_size_kb: Maximum file size in KB to include

    Returns:
        DirectoryInfo representing the tree structure
    """
    return _build_tree_recursive(
        root, root, max_depth, max_files_per_dir, max_file_size_kb
    )


def _build_tree_recursive(
    root: Path,
    current: Path,
    max_depth: int,
    max_files: int,
    max_size_kb: int,
    depth: int = 0,
) -> DirectoryInfo:
    """Recursively build directory tree."""
    # Collect files
    files: list[FileInfo] = []
    try:
        for item in sorted(current.iterdir()):
            if should_ignore(item):
                continue

            if item.is_file():
                # Check file size
                try:
                    size_bytes = item.stat().st_size
                    if size_bytes > max_size_kb * 1024:
                        continue
                except OSError:
                    continue

                # Categorize and add file
                category = categorize_file(item)
                relative = str(item.relative_to(root))
                files.append(
                    FileInfo(
                        path=item,
                        relative_path=relative,
                        size_bytes=size_bytes,
                        category=category,
                    )
                )

                # Limit files per directory
                if len(files) >= max_files:
                    break
    except PermissionError:
        pass

    # Collect subdirectories
    subdirs: list[DirectoryInfo] = []
    if depth < max_depth:
        try:
            for item in sorted(current.iterdir()):
                if should_ignore(item):
                    continue

                if item.is_dir():
                    subdir = _build_tree_recursive(
                        root, item, max_depth, max_files, max_size_kb, depth + 1
                    )
                    # Only include non-empty directories
                    if subdir.files or subdir.subdirs:
                        subdirs.append(subdir)
        except PermissionError:
            pass

    relative = str(current.relative_to(root)) if current != root else "."
    return DirectoryInfo(
        path=current,
        relative_path=relative,
        files=tuple(files),
        subdirs=tuple(subdirs),
    )


def render_file_tree(
    root: Path,
    max_depth: int = 3,
    max_files_per_dir: int = 50,
    max_file_size_kb: int = 1024,
) -> str:
    """
    Render file tree as string.

    Args:
        root: Root directory path
        max_depth: Maximum depth to traverse
        max_files_per_dir: Maximum files per directory
        max_file_size_kb: Maximum file size in KB

    Returns:
        Formatted tree string
    """
    tree = build_file_tree(root, max_depth, max_files_per_dir, max_file_size_kb)

    lines = [
        f"Project Structure:",
        f"Total Files: {tree.count_files()}",
        f"Total Directories: {tree.count_dirs()}",
        "",
        tree.to_tree_string(show_files=True),
    ]

    return "\n".join(lines)
