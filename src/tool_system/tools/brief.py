from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class BriefTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="BriefPreview",
            description="Create a brief summary/preview of text content.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"text": {"type": "string"}, "max_chars": {"type": "integer"}},
                "required": ["text"],
            },
            is_read_only=True,
            max_result_size_chars=50_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        text = tool_input["text"]
        if not isinstance(text, str):
            raise ToolInputError("text must be a string")
        max_chars = tool_input.get("max_chars", 1000)
        if not isinstance(max_chars, int) or max_chars < 1 or max_chars > 50_000:
            raise ToolInputError("max_chars must be an integer between 1 and 50000")
        preview = text if len(text) <= max_chars else text[:max_chars] + "…"
        return ToolResult(
            name="BriefPreview",
            output={"preview": preview, "original_chars": len(text), "max_chars": max_chars},
        )
