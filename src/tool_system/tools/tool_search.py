from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolRegistry, ToolSpec


class ToolSearchTool:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ToolSearch",
            description="Search for available tools by name or keywords.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query must be a non-empty string")
        max_results = tool_input.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 50:
            raise ToolInputError("max_results must be an integer between 1 and 50")

        q = query.strip()
        lowered = q.lower()
        if lowered.startswith("select:"):
            name = q.split(":", 1)[1].strip()
            tool = self._registry.get(name)
            matches = [tool.spec().name] if tool else []
            return ToolResult(
                name="ToolSearch",
                output={
                    "matches": matches,
                    "query": query,
                    "total_deferred_tools": 0,
                },
            )

        scored: list[tuple[int, str]] = []
        for spec in self._registry.list_specs():
            hay = f"{spec.name}\n{spec.description}".lower()
            if lowered in spec.name.lower():
                scored.append((0, spec.name))
            elif lowered in hay:
                scored.append((1, spec.name))
        scored.sort(key=lambda t: (t[0], t[1].lower()))
        matches = [name for _, name in scored[:max_results]]
        return ToolResult(
            name="ToolSearch",
            output={"matches": matches, "query": query, "total_deferred_tools": 0},
        )

