from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .errors import ToolPermissionError


def _resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class ToolPermissionContext:
    deny_names: frozenset[str] = field(default_factory=frozenset)
    deny_prefixes: tuple[str, ...] = ()
    workspace_root: Path | None = None
    additional_working_directories: tuple[Path, ...] = ()
    allow_docs: bool = False

    @classmethod
    def from_iterables(
        cls,
        deny_names: Iterable[str] | None = None,
        deny_prefixes: Iterable[str] | None = None,
        *,
        workspace_root: str | Path | None = None,
        additional_working_directories: Iterable[str | Path] | None = None,
        allow_docs: bool = False,
    ) -> "ToolPermissionContext":
        return cls(
            deny_names=frozenset(name.lower() for name in (deny_names or [])),
            deny_prefixes=tuple(prefix.lower() for prefix in (deny_prefixes or [])),
            workspace_root=_resolve_path(workspace_root) if workspace_root else None,
            additional_working_directories=tuple(
                _resolve_path(p) for p in (additional_working_directories or [])
            ),
            allow_docs=allow_docs,
        )

    def blocks_tool(self, tool_name: str) -> bool:
        lowered = tool_name.lower()
        return lowered in self.deny_names or any(
            lowered.startswith(prefix) for prefix in self.deny_prefixes
        )

    def allowed_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        if self.workspace_root is not None:
            roots.append(self.workspace_root)
        roots.extend(self.additional_working_directories)
        return tuple(roots)

    def ensure_path_allowed(self, path: str | Path) -> Path:
        resolved = _resolve_path(path)
        roots = self.allowed_roots()
        if not roots:
            return resolved
        if any(_is_within(resolved, root) for root in roots):
            return resolved
        roots_str = ", ".join(str(r) for r in roots)
        raise ToolPermissionError(f"path is outside allowed working directories: {resolved} (allowed: {roots_str})")

