from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..protocol import ToolResult
from ..registry import ToolSpec


class StructuredOutputTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="StructuredOutput",
            description="Return a final response as structured JSON.",
            input_schema={"type": "object", "additionalProperties": True},
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        context.outbox.append({"tool": "StructuredOutput", "structured_output": tool_input})
        return ToolResult(
            name="StructuredOutput",
            output={
                "data": "Structured output provided successfully",
                "structured_output": tool_input,
            },
        )

