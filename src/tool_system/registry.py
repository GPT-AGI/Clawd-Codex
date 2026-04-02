from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from .context import ToolContext
from .protocol import ToolCall, ToolResult
from .schema_validation import validate_json_schema


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    aliases: tuple[str, ...] = ()
    is_read_only: bool = False
    is_destructive: bool = False
    strict: bool = False
    max_result_size_chars: int = 20_000


class Tool(Protocol):
    def spec(self) -> ToolSpec: ...

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult: ...


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: list[Tool] = []
        self._by_name: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        spec = tool.spec()
        key = spec.name.lower()
        if key in self._by_name:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._tools.append(tool)
        self._by_name[key] = tool
        for alias in spec.aliases:
            alias_key = alias.lower()
            if alias_key in self._by_name:
                raise ValueError(f"duplicate tool alias: {alias}")
            self._by_name[alias_key] = tool

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools]

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name.lower())

    def dispatch(self, call: ToolCall, context: ToolContext) -> ToolResult:
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                name=call.name,
                output={"error": f"unknown tool: {call.name}"},
                is_error=True,
                tool_use_id=call.tool_use_id,
            )
        spec = tool.spec()
        context.ensure_tool_allowed(spec.name)
        validate_json_schema(call.input, spec.input_schema, root_name=spec.name)
        result = tool.run(call.input, context)
        if result.tool_use_id is None and call.tool_use_id is not None:
            return ToolResult(
                name=result.name,
                output=result.output,
                is_error=result.is_error,
                tool_use_id=call.tool_use_id,
                content_type=result.content_type,
            )
        return result

