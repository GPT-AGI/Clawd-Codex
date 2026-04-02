from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


_VCS_DIRS = {".git", ".svn", ".hg", ".bzr", ".jj", ".sl"}


def _iter_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _VCS_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def _matches_glob(path: Path, pattern: str) -> bool:
    return fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern)


def _matches_type(path: Path, type_name: str) -> bool:
    ext = path.suffix.lower().lstrip(".")
    if not ext:
        return False
    return ext == type_name.lower()


@dataclass(frozen=True)
class _Pagination:
    items: list[Any]
    applied_limit: int | None
    applied_offset: int


def _paginate(items: list[Any], *, head_limit: int | None, offset: int) -> _Pagination:
    if head_limit == 0:
        return _Pagination(items=items[offset:], applied_limit=None, applied_offset=offset)
    effective_limit = head_limit if head_limit is not None else 250
    sliced = items[offset : offset + effective_limit]
    truncated = len(items) - offset > effective_limit
    return _Pagination(items=sliced, applied_limit=effective_limit if truncated else None, applied_offset=offset)


class GrepTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Grep",
            description="A powerful search tool built on regex search.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                    },
                    "-B": {"type": "integer"},
                    "-A": {"type": "integer"},
                    "-C": {"type": "integer"},
                    "context": {"type": "integer"},
                    "-n": {"type": "boolean"},
                    "-i": {"type": "boolean"},
                    "type": {"type": "string"},
                    "head_limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                    "multiline": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
            is_read_only=True,
            strict=True,
            max_result_size_chars=20_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = tool_input["pattern"]
        if not isinstance(pattern, str) or pattern == "":
            raise ToolInputError("pattern must be a non-empty string")

        base = tool_input.get("path")
        glob_pattern = tool_input.get("glob")
        type_name = tool_input.get("type")
        output_mode = tool_input.get("output_mode", "files_with_matches")
        if output_mode not in {"content", "files_with_matches", "count"}:
            raise ToolInputError("invalid output_mode")

        head_limit = tool_input.get("head_limit")
        offset = tool_input.get("offset", 0)
        if head_limit is not None and (not isinstance(head_limit, int) or head_limit < 0):
            raise ToolInputError("head_limit must be an integer >= 0")
        if not isinstance(offset, int) or offset < 0:
            raise ToolInputError("offset must be an integer >= 0")

        case_insensitive = bool(tool_input.get("-i", False))
        multiline = bool(tool_input.get("multiline", False))
        show_line_numbers = bool(tool_input.get("-n", False)) if output_mode == "content" else False

        flags = re.MULTILINE
        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.DOTALL
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise ToolInputError(f"invalid regex: {e}") from e

        base_path = context.cwd if base is None else context.ensure_allowed_path(base)
        if not base_path.exists():
            raise ToolInputError(f"path does not exist: {base_path}")

        files_to_search: list[Path] = []
        if base_path.is_file():
            files_to_search = [base_path]
        else:
            files_to_search = [p for p in _iter_files(base_path) if p.is_file()]

        if glob_pattern:
            files_to_search = [p for p in files_to_search if _matches_glob(p, glob_pattern)]
        if type_name:
            files_to_search = [p for p in files_to_search if _matches_type(p, type_name)]

        matched_files: list[Path] = []
        content_lines: list[str] = []
        total_matches = 0

        for file in files_to_search:
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if regex.search(text) is None:
                continue
            matched_files.append(file)

            if output_mode == "content":
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line) is None:
                        continue
                    total_matches += len(list(regex.finditer(line)))
                    prefix = f"{file}:{line_no}:" if show_line_numbers else f"{file}:"
                    content_lines.append(prefix + line)
            elif output_mode == "count":
                matches = len(list(regex.finditer(text)))
                total_matches += matches

        if output_mode == "content":
            paged = _paginate(content_lines, head_limit=head_limit, offset=offset)
            output: dict[str, Any] = {
                "mode": "content",
                "numFiles": len(matched_files),
                "filenames": [str(p) for p in matched_files],
                "content": "\n".join(paged.items),
                "numLines": len(paged.items),
                "appliedOffset": paged.applied_offset,
            }
            if paged.applied_limit is not None:
                output["appliedLimit"] = paged.applied_limit
            return ToolResult(name="Grep", output=output)

        if output_mode == "count":
            filenames = [str(p) for p in matched_files]
            paged = _paginate(filenames, head_limit=head_limit, offset=offset)
            output = {
                "mode": "count",
                "numFiles": len(matched_files),
                "filenames": paged.items,
                "numMatches": total_matches,
                "appliedOffset": paged.applied_offset,
            }
            if paged.applied_limit is not None:
                output["appliedLimit"] = paged.applied_limit
            return ToolResult(name="Grep", output=output)

        filenames = [str(p) for p in matched_files]
        paged = _paginate(filenames, head_limit=head_limit, offset=offset)
        output = {
            "mode": "files_with_matches",
            "numFiles": len(matched_files),
            "filenames": paged.items,
            "appliedOffset": paged.applied_offset,
        }
        if paged.applied_limit is not None:
            output["appliedLimit"] = paged.applied_limit
        return ToolResult(name="Grep", output=output)

