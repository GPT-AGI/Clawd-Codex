from __future__ import annotations

import glob as globlib
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class GlobTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Glob",
            description=(
                "- Fast file pattern matching tool that works with any codebase size\n"
                '- Supports glob patterns like "**/*.js" or "src/**/*.ts"\n'
                "- Returns matching file paths sorted by modification time"
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["pattern"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = tool_input["pattern"]
        base = tool_input.get("path")
        limit = tool_input.get("limit", 100)
        if not isinstance(pattern, str) or not pattern:
            raise ToolInputError("pattern must be a non-empty string")
        if base is not None and (not isinstance(base, str) or not base):
            raise ToolInputError("path must be a non-empty string when provided")
        if not isinstance(limit, int) or limit < 1 or limit > 10_000:
            raise ToolInputError("limit must be an integer between 1 and 10000")

        base_dir = context.cwd if base is None else context.ensure_allowed_path(base)
        if not base_dir.exists():
            raise ToolInputError(f"path does not exist: {base_dir}")
        if not base_dir.is_dir():
            raise ToolInputError(f"path is not a directory: {base_dir}")

        full_pattern = str(base_dir / pattern)
        matches = [Path(p) for p in globlib.glob(full_pattern, recursive=True)]
        files = [p for p in matches if p.is_file()]

        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        truncated = len(files) > limit
        files = files[:limit]
        return ToolResult(
            name="Glob",
            output={
                "filenames": [str(p) for p in files],
                "numFiles": len(files),
                "truncated": truncated,
            },
        )

