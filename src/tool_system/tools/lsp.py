from __future__ import annotations

from typing import Any, Protocol

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class LSPClient(Protocol):
    def request(self, method: str, params: dict[str, Any] | None = None) -> Any: ...


class LSPTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="LSP",
            description="Send a request to the configured Language Server Protocol client.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"method": {"type": "string"}, "params": {"type": "object"}},
                "required": ["method"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        method = tool_input["method"]
        params = tool_input.get("params")
        if not isinstance(method, str) or not method:
            raise ToolInputError("method must be a non-empty string")
        if params is not None and not isinstance(params, dict):
            raise ToolInputError("params must be an object when provided")

        client = context.lsp_client
        if client is None:
            return ToolResult(name="LSP", output={"error": "no lsp client configured"}, is_error=True)

        out = client.request(method, params)
        return ToolResult(name="LSP", output={"method": method, "response": out})

