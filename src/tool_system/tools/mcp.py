from __future__ import annotations

from typing import Any, Protocol

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class MCPClient(Protocol):
    def call_tool(self, tool_name: str, args: dict[str, Any]) -> Any: ...

    def list_tools(self) -> list[str]: ...


class MCPTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="MCP",
            description="Call a tool exposed by a connected MCP server.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "server": {"type": "string"},
                    "tool": {"type": "string"},
                    "input": {"type": "object"},
                },
                "required": ["server", "tool"],
            },
            is_destructive=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        server = tool_input["server"]
        tool_name = tool_input["tool"]
        args = tool_input.get("input") or {}
        if not isinstance(server, str) or not server:
            raise ToolInputError("server must be a non-empty string")
        if not isinstance(tool_name, str) or not tool_name:
            raise ToolInputError("tool must be a non-empty string")
        if not isinstance(args, dict):
            raise ToolInputError("input must be an object when provided")

        client = context.mcp_clients.get(server)
        if client is None:
            return ToolResult(name="MCP", output={"error": f"mcp server not connected: {server}"}, is_error=True)

        out = client.call_tool(tool_name, args)
        return ToolResult(name="MCP", output={"server": server, "tool": tool_name, "output": out})

