from __future__ import annotations

import uuid
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


_TASK_STATUSES = {"pending", "in_progress", "completed"}


def _new_task_id() -> str:
    return uuid.uuid4().hex[:12]


class TaskCreateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskCreate",
            description="Create a task in the task list.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "activeForm": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["subject", "description"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        subject = tool_input.get("subject")
        description = tool_input.get("description")
        active_form = tool_input.get("activeForm") or ""
        metadata = tool_input.get("metadata") or {}
        if not isinstance(subject, str) or not subject.strip():
            raise ToolInputError("subject must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ToolInputError("description must be a non-empty string")
        if not isinstance(active_form, str):
            raise ToolInputError("activeForm must be a string when provided")
        if not isinstance(metadata, dict):
            raise ToolInputError("metadata must be an object when provided")

        task_id = _new_task_id()
        context.tasks[task_id] = {
            "id": task_id,
            "subject": subject,
            "description": description,
            "activeForm": active_form,
            "status": "pending",
            "owner": None,
            "blocks": [],
            "blockedBy": [],
            "metadata": dict(metadata),
            "output": "",
        }
        return ToolResult(name="TaskCreate", output={"task": {"id": task_id, "subject": subject}})


class TaskGetTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskGet",
            description="Retrieve a task by ID.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"taskId": {"type": "string"}},
                "required": ["taskId"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = tool_input.get("taskId")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ToolInputError("taskId must be a non-empty string")
        task = context.tasks.get(task_id)
        if task is None:
            return ToolResult(name="TaskGet", output={"task": None})
        return ToolResult(
            name="TaskGet",
            output={
                "task": {
                    "id": task["id"],
                    "subject": task["subject"],
                    "description": task["description"],
                    "status": task["status"],
                    "blocks": list(task.get("blocks") or []),
                    "blockedBy": list(task.get("blockedBy") or []),
                }
            },
        )


class TaskListTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskList",
            description="List all tasks.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        tasks = []
        for t in context.tasks.values():
            tasks.append(
                {
                    "id": t["id"],
                    "subject": t["subject"],
                    "status": t["status"],
                    **({"owner": t["owner"]} if t.get("owner") else {}),
                    "blockedBy": list(t.get("blockedBy") or []),
                }
            )
        tasks.sort(key=lambda x: x["id"])
        return ToolResult(name="TaskList", output={"tasks": tasks})


class TaskUpdateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskUpdate",
            description="Update a task.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "taskId": {"type": "string"},
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "activeForm": {"type": "string"},
                    "status": {"type": "string"},
                    "addBlocks": {"type": "array", "items": {"type": "string"}},
                    "addBlockedBy": {"type": "array", "items": {"type": "string"}},
                    "owner": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["taskId"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = tool_input.get("taskId")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ToolInputError("taskId must be a non-empty string")
        task = context.tasks.get(task_id)
        if task is None:
            return ToolResult(
                name="TaskUpdate",
                output={"success": False, "taskId": task_id, "updatedFields": [], "error": "Task not found"},
            )

        updated_fields: list[str] = []
        status_change: dict[str, str] | None = None

        for field in ("subject", "description", "activeForm", "owner"):
            if field in tool_input and tool_input[field] is not None:
                v = tool_input[field]
                if not isinstance(v, str):
                    raise ToolInputError(f"{field} must be a string when provided")
                if v != task.get(field):
                    task[field] = v
                    updated_fields.append(field)

        if "status" in tool_input and tool_input["status"] is not None:
            status = tool_input["status"]
            if not isinstance(status, str) or status not in _TASK_STATUSES and status != "deleted":
                raise ToolInputError("status must be pending|in_progress|completed|deleted when provided")
            if status == "deleted":
                context.tasks.pop(task_id, None)
                return ToolResult(
                    name="TaskUpdate",
                    output={"success": True, "taskId": task_id, "updatedFields": ["deleted"]},
                )
            if status != task.get("status"):
                status_change = {"from": str(task.get("status")), "to": status}
                task["status"] = status
                updated_fields.append("status")

        for rel_field, input_key in (("blocks", "addBlocks"), ("blockedBy", "addBlockedBy")):
            if input_key in tool_input and tool_input[input_key] is not None:
                ids = tool_input[input_key]
                if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                    raise ToolInputError(f"{input_key} must be an array of strings when provided")
                cur = list(task.get(rel_field) or [])
                for x in ids:
                    if x not in cur:
                        cur.append(x)
                if cur != task.get(rel_field):
                    task[rel_field] = cur
                    updated_fields.append(rel_field)

        if "metadata" in tool_input and tool_input["metadata"] is not None:
            md = tool_input["metadata"]
            if not isinstance(md, dict):
                raise ToolInputError("metadata must be an object when provided")
            existing = dict(task.get("metadata") or {})
            for k, v in md.items():
                if v is None:
                    existing.pop(k, None)
                else:
                    existing[k] = v
            task["metadata"] = existing
            updated_fields.append("metadata")

        out: dict[str, Any] = {"success": True, "taskId": task_id, "updatedFields": updated_fields}
        if status_change is not None:
            out["statusChange"] = status_change
        return ToolResult(name="TaskUpdate", output=out)


class TaskOutputTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TaskOutput",
            description="Get output for a task (best-effort).",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "task_id": {"type": "string"},
                    "block": {"type": "boolean"},
                    "timeout": {"type": "number"},
                },
                "required": ["task_id"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
            aliases=("AgentOutputTool", "BashOutputTool"),
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = tool_input.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ToolInputError("task_id must be a non-empty string")

        task = context.tasks.get(task_id)
        if task is None:
            return ToolResult(name="TaskOutput", output={"retrieval_status": "success", "task": None})

        output = str(task.get("output") or "")
        retrieval_status = "success" if output else "not_ready"
        return ToolResult(
            name="TaskOutput",
            output={
                "retrieval_status": retrieval_status,
                "task": {
                    "task_id": task_id,
                    "task_type": "task_list",
                    "status": task.get("status"),
                    "description": task.get("description"),
                    "output": output,
                },
            },
        )

