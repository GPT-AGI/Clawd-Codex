from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolCall, ToolResult
from ..registry import ToolRegistry, ToolSpec


class AgentTool:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Agent",
            description="Execute a sequence of tool calls as a single atomic agent step.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {"name": {"type": "string"}, "input": {"type": "object"}},
                            "required": ["name", "input"],
                        },
                    },
                    "stop_on_error": {"type": "boolean"},
                },
                "required": ["calls"],
            },
            aliases=("Task",),
            is_destructive=True,
            max_result_size_chars=200_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        calls = tool_input["calls"]
        stop_on_error = bool(tool_input.get("stop_on_error", True))
        if not isinstance(calls, list):
            raise ToolInputError("calls must be an array")

        results: list[dict[str, Any]] = []
        any_error = False
        for idx, call in enumerate(calls):
            if not isinstance(call, dict):
                raise ToolInputError(f"calls[{idx}] must be an object")
            name = call.get("name")
            inp = call.get("input")
            if not isinstance(name, str) or not isinstance(inp, dict):
                raise ToolInputError(f"calls[{idx}] must include name:string and input:object")
            result = self._registry.dispatch(ToolCall(name=name, input=inp), context)
            results.append({"name": name, "is_error": result.is_error, "output": result.output})
            any_error = any_error or result.is_error
            if result.is_error and stop_on_error:
                break

        return ToolResult(name="Agent", output={"results": results}, is_error=any_error)
