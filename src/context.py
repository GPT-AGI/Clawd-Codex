from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .file_tree import DirectoryInfo, build_file_tree
from .git_context import GitStatus, build_git_status


@dataclass(frozen=True)
class PortContext:
    source_root: Path
    tests_root: Path
    assets_root: Path
    archive_root: Path
    python_file_count: int
    test_file_count: int
    asset_file_count: int
    archive_available: bool
    git_status: GitStatus | None = None
    file_tree: DirectoryInfo | None = None


def build_port_context(
    base: Path | None = None,
    include_git: bool = True,
    include_tree: bool = True,
    tree_max_depth: int = 3,
) -> PortContext:
    """
    Build comprehensive workspace context.

    Args:
        base: Base directory path (defaults to project root)
        include_git: Whether to include git status
        include_tree: Whether to include file tree
        tree_max_depth: Maximum depth for file tree

    Returns:
        PortContext with workspace information
    """
    root = base or Path(__file__).resolve().parent.parent
    source_root = root / 'src'
    tests_root = root / 'tests'
    assets_root = root / 'assets'
    archive_root = root / 'archive' / 'claude_code_ts_snapshot' / 'src'

    # Build git status
    git_status = None
    if include_git:
        try:
            git_status = build_git_status(root)
        except Exception:
            pass  # Ignore git errors

    # Build file tree
    file_tree = None
    if include_tree:
        try:
            file_tree = build_file_tree(root, max_depth=tree_max_depth)
        except Exception:
            pass  # Ignore tree errors

    return PortContext(
        source_root=source_root,
        tests_root=tests_root,
        assets_root=assets_root,
        archive_root=archive_root,
        python_file_count=sum(1 for path in source_root.rglob('*.py') if path.is_file()),
        test_file_count=sum(1 for path in tests_root.rglob('*.py') if path.is_file()),
        asset_file_count=sum(1 for path in assets_root.rglob('*') if path.is_file()),
        archive_available=archive_root.exists(),
        git_status=git_status,
        file_tree=file_tree,
    )


def render_context(context: PortContext, include_tree: bool = False) -> str:
    """
    Render workspace context as human-readable string.

    Args:
        context: PortContext to render
        include_tree: Whether to include full file tree

    Returns:
        Formatted context string
    """
    lines = [
        f'Source root: {context.source_root}',
        f'Test root: {context.tests_root}',
        f'Assets root: {context.assets_root}',
        f'Archive root: {context.archive_root}',
        f'Python files: {context.python_file_count}',
        f'Test files: {context.test_file_count}',
        f'Assets: {context.asset_file_count}',
        f'Archive available: {context.archive_available}',
    ]

    # Add git status
    if context.git_status:
        lines.append('')
        lines.append(context.git_status.to_context_string())

    # Add file tree summary
    if context.file_tree:
        lines.append('')
        lines.append(f'File Tree Summary:')
        lines.append(f'  Total Files: {context.file_tree.count_files()}')
        lines.append(f'  Total Directories: {context.file_tree.count_dirs()}')

        # Optionally include full tree
        if include_tree:
            lines.append('')
            lines.append(context.file_tree.to_tree_string(show_files=True))

    return '\n'.join(lines)
