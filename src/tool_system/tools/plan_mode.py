from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..protocol import ToolResult
from ..registry import ToolSpec


class EnterPlanModeTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="EnterPlanMode",
            description="Enter plan mode (exploration/planning phase).",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        context.plan_mode = True
        return ToolResult(
            name="EnterPlanMode",
            output={
                "message": "Entered plan mode. Explore and design an approach before editing files.",
            },
        )


class ExitPlanModeTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ExitPlanMode",
            description="Exit plan mode after writing a plan; may persist plan to disk.",
            input_schema={
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "allowedPrompts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "tool": {"type": "string", "enum": ["Bash"]},
                                "prompt": {"type": "string"},
                            },
                            "required": ["tool", "prompt"],
                        },
                    },
                    "plan": {"type": "string"},
                    "planFilePath": {"type": "string"},
                },
            },
            is_destructive=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if not context.plan_mode:
            raise ToolPermissionError("not in plan mode")

        plan = tool_input.get("plan")
        if plan is not None and not isinstance(plan, str):
            raise ToolInputError("plan must be a string when provided")
        plan_path = tool_input.get("planFilePath")
        if plan_path is not None and not isinstance(plan_path, str):
            raise ToolInputError("planFilePath must be a string when provided")

        file_path_out: str | None = None
        if plan is not None:
            if plan_path:
                target = Path(plan_path).expanduser()
                if not target.is_absolute():
                    target = (context.cwd or context.workspace_root) / target
                target = context.ensure_allowed_path(target)
            else:
                target = context.workspace_root / ".clawd" / "plan.md"
                target.parent.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(plan, encoding="utf-8")
            file_path_out = str(target)

        context.plan_mode = False
        out: dict[str, Any] = {
            "plan": plan if plan is not None else None,
            "isAgent": False,
            "filePath": file_path_out,
            "hasTaskTool": True,
            "planWasEdited": False,
            "requestId": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        }
        return ToolResult(name="ExitPlanMode", output=out)

