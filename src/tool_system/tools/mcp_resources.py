from __future__ import annotations

from typing import Any, Protocol

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class _McpResourceClient(Protocol):
    def list_resources(self) -> list[dict[str, Any]]: ...

    def read_resource(self, uri: str) -> dict[str, Any]: ...


class ListMcpResourcesTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ListMcpResourcesTool",
            description="List resources from connected MCP servers.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"server": {"type": "string"}},
            },
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        server = tool_input.get("server")
        if server is not None and (not isinstance(server, str) or not server.strip()):
            raise ToolInputError("server must be a non-empty string when provided")

        clients: list[tuple[str, Any]]
        if server:
            client = context.mcp_clients.get(server)
            if client is None:
                return ToolResult(name="ListMcpResourcesTool", output={"error": f"mcp server not connected: {server}"}, is_error=True)
            clients = [(server, client)]
        else:
            clients = list(context.mcp_clients.items())

        resources: list[dict[str, Any]] = []
        for name, client in clients:
            if hasattr(client, "list_resources"):
                try:
                    items = client.list_resources()
                except Exception as e:
                    resources.append({"server": name, "uri": "", "name": "", "description": str(e)})
                    continue
                if isinstance(items, list):
                    for r in items:
                        if isinstance(r, dict):
                            resources.append(
                                {
                                    "uri": str(r.get("uri", "")),
                                    "name": str(r.get("name", "")),
                                    "mimeType": r.get("mimeType"),
                                    "description": r.get("description"),
                                    "server": name,
                                }
                            )
        return ToolResult(name="ListMcpResourcesTool", output=resources)


class ReadMcpResourceTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ReadMcpResourceTool",
            description="Read a specific MCP resource by URI.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"server": {"type": "string"}, "uri": {"type": "string"}},
                "required": ["server", "uri"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        server = tool_input.get("server")
        uri = tool_input.get("uri")
        if not isinstance(server, str) or not server.strip():
            raise ToolInputError("server must be a non-empty string")
        if not isinstance(uri, str) or not uri.strip():
            raise ToolInputError("uri must be a non-empty string")

        client = context.mcp_clients.get(server)
        if client is None:
            return ToolResult(name="ReadMcpResourceTool", output={"error": f"mcp server not connected: {server}"}, is_error=True)
        if not hasattr(client, "read_resource"):
            return ToolResult(name="ReadMcpResourceTool", output={"error": f"mcp server does not support resources: {server}"}, is_error=True)
        out = client.read_resource(uri)
        if isinstance(out, dict) and "contents" in out:
            return ToolResult(name="ReadMcpResourceTool", output=out)
        return ToolResult(name="ReadMcpResourceTool", output={"contents": [{"uri": uri, **(out if isinstance(out, dict) else {"text": str(out)})}]})

