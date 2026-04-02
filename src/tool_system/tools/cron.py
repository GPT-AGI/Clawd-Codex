from __future__ import annotations

import uuid
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class CronCreateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="CronCreate",
            description="Schedule a recurring or one-shot prompt (in-memory).",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "cron": {"type": "string"},
                    "prompt": {"type": "string"},
                    "recurring": {"type": "boolean"},
                    "durable": {"type": "boolean"},
                },
                "required": ["cron", "prompt"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        cron = tool_input.get("cron")
        prompt = tool_input.get("prompt")
        if not isinstance(cron, str) or not cron.strip():
            raise ToolInputError("cron must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ToolInputError("prompt must be a non-empty string")
        recurring = bool(tool_input.get("recurring", True))
        durable = bool(tool_input.get("durable", False))

        cid = uuid.uuid4().hex[:12]
        context.crons[cid] = {"id": cid, "cron": cron, "prompt": prompt, "recurring": recurring, "durable": durable}
        return ToolResult(
            name="CronCreate",
            output={
                "id": cid,
                "humanSchedule": cron,
                "recurring": recurring,
                "durable": durable,
            },
        )


class CronListTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="CronList",
            description="List scheduled cron jobs.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        jobs = list(context.crons.values())
        jobs.sort(key=lambda x: x["id"])
        return ToolResult(name="CronList", output={"jobs": jobs})


class CronDeleteTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="CronDelete",
            description="Delete a scheduled cron job.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        cid = tool_input.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise ToolInputError("id must be a non-empty string")
        existed = cid in context.crons
        context.crons.pop(cid, None)
        return ToolResult(name="CronDelete", output={"success": existed, "id": cid})

