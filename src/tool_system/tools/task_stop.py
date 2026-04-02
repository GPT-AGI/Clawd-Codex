from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class TaskStopTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskStop",
            description="Stop a previously started background task by id.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
            is_destructive=True,
            max_result_size_chars=2000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = tool_input["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise ToolInputError("task_id must be a non-empty string")
        stopped = context.task_manager.stop(task_id)
        return ToolResult(name="TaskStop", output={"task_id": task_id, "stopped": stopped}, is_error=not stopped)

