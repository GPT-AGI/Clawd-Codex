from __future__ import annotations

import time
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class SleepTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Sleep",
            description="Sleep for a short duration.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"seconds": {"type": "number"}},
                "required": ["seconds"],
            },
            is_read_only=True,
            max_result_size_chars=1000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        seconds = tool_input["seconds"]
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            raise ToolInputError("seconds must be a number")
        if seconds < 0 or seconds > 30:
            raise ToolInputError("seconds must be between 0 and 30")
        time.sleep(float(seconds))
        return ToolResult(name="Sleep", output={"slept_seconds": float(seconds)})

