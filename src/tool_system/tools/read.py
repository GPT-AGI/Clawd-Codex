from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolExecutionError, ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class FileReadTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Read",
            description="Read a file from the local filesystem.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file_path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "pages": {"type": "string"},
                },
                "required": ["file_path"],
            },
            is_read_only=True,
            max_result_size_chars=1_000_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = tool_input["file_path"]
        if not isinstance(file_path, str):
            raise ToolInputError("file_path must be a string")

        limit = tool_input.get("limit")
        offset = tool_input.get("offset")
        pages = tool_input.get("pages")
        if limit is None:
            limit = 2000
        if offset is None:
            offset = 1
        if not isinstance(limit, int) or limit < 1 or limit > 2000:
            raise ToolInputError("limit must be an integer between 1 and 2000")
        if not isinstance(offset, int) or offset < 1:
            raise ToolInputError("offset must be an integer >= 1")
        if pages is not None and not isinstance(pages, str):
            raise ToolInputError("pages must be a string when provided")

        path = context.ensure_allowed_path(file_path)
        if not path.exists():
            return ToolResult(name="Read", output={"error": f"file not found: {path}"}, is_error=True)
        if path.is_dir():
            return ToolResult(name="Read", output={"error": f"path is a directory: {path}"}, is_error=True)

        if self._is_blocked_device_path(path):
            return ToolResult(name="Read", output={"error": f"blocked device path: {path}"}, is_error=True)

        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return self._read_image(path, context)
        if suffix == ".pdf":
            return self._read_pdf(path, context, pages=pages)
        if suffix == ".ipynb":
            return self._read_notebook(path, context)

        if context.was_file_read_and_unchanged(path) and pages is None and tool_input.get("offset") is None and tool_input.get("limit") is None:
            return ToolResult(name="Read", output={"type": "file_unchanged", "file": {"filePath": str(path)}})

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise ToolExecutionError(str(e)) from e

        lines = text.splitlines()
        start = offset - 1
        end = start + limit
        sliced = lines[start:end]
        numbered = "\n".join(f"{i + offset}\t{line}" for i, line in enumerate(sliced))
        if numbered == "" and text == "":
            numbered = ""
        context.mark_file_read(path)
        return ToolResult(
            name="Read",
            output={
                "type": "text",
                "file": {
                    "filePath": str(path),
                    "content": numbered,
                    "numLines": len(sliced),
                    "startLine": offset,
                    "totalLines": len(lines),
                },
            },
        )

    def _read_image(self, path: Path, context: ToolContext) -> ToolResult:
        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        data = path.read_bytes()
        if len(data) > 5 * 1024 * 1024:
            return ToolResult(
                name="Read",
                output={"error": f"image too large to inline: {path} ({len(data)} bytes)"},
                is_error=True,
            )
        encoded = base64.b64encode(data).decode("ascii")
        context.mark_file_read(path)
        return ToolResult(
            name="Read",
            output={
                "type": "image",
                "file": {
                    "base64": encoded,
                    "type": mime,
                    "originalSize": len(data),
                    "filePath": str(path),
                },
            },
        )

    def _read_pdf(self, path: Path, context: ToolContext, *, pages: str | None) -> ToolResult:
        if pages is not None and pages.strip():
            return ToolResult(name="Read", output={"error": "PDF page-range reads are not supported in this build; omit pages to read the full PDF"}, is_error=True)
        data = path.read_bytes()
        if len(data) > 5 * 1024 * 1024:
            return ToolResult(name="Read", output={"error": f"pdf too large to inline: {path} ({len(data)} bytes)"}, is_error=True)
        encoded = base64.b64encode(data).decode("ascii")
        context.mark_file_read(path)
        return ToolResult(
            name="Read",
            output={"type": "pdf", "file": {"filePath": str(path), "base64": encoded, "originalSize": len(data)}},
        )

    def _read_notebook(self, path: Path, context: ToolContext) -> ToolResult:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
        except Exception as e:
            return ToolResult(name="Read", output={"error": f"failed to parse notebook: {e}"}, is_error=True)
        cells = data.get("cells")
        if not isinstance(cells, list):
            return ToolResult(name="Read", output={"error": "no cells found in notebook"}, is_error=True)
        context.mark_file_read(path)
        return ToolResult(name="Read", output={"type": "notebook", "file": {"filePath": str(path), "cells": cells}})

    def _is_blocked_device_path(self, path: Path) -> bool:
        p = str(path)
        blocked = {
            "/dev/zero",
            "/dev/random",
            "/dev/urandom",
            "/dev/full",
            "/dev/stdin",
            "/dev/tty",
            "/dev/console",
            "/dev/stdout",
            "/dev/stderr",
            "/dev/fd/0",
            "/dev/fd/1",
            "/dev/fd/2",
        }
        if p in blocked:
            return True
        if p.startswith("/proc/") and (p.endswith("/fd/0") or p.endswith("/fd/1") or p.endswith("/fd/2")):
            return True
        return False
