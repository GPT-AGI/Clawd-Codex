from __future__ import annotations

import json
import uuid
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class TeamCreateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamCreate",
            description="Create a lightweight team context for multi-agent workflows.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "team_name": {"type": "string"},
                    "description": {"type": "string"},
                    "agent_type": {"type": "string"},
                },
                "required": ["team_name"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        team_name = tool_input.get("team_name")
        if not isinstance(team_name, str) or not team_name.strip():
            raise ToolInputError("team_name must be a non-empty string")
        description = tool_input.get("description")
        if description is not None and not isinstance(description, str):
            raise ToolInputError("description must be a string when provided")
        agent_type = tool_input.get("agent_type")
        if agent_type is not None and not isinstance(agent_type, str):
            raise ToolInputError("agent_type must be a string when provided")

        lead_agent_id = uuid.uuid4().hex[:12]
        team_file = context.workspace_root / ".clawd" / "team.json"
        team_file.parent.mkdir(parents=True, exist_ok=True)
        team = {"team_name": team_name, "description": description, "agent_type": agent_type, "lead_agent_id": lead_agent_id}
        team_file.write_text(json.dumps(team, ensure_ascii=False, indent=2), encoding="utf-8")
        context.team = team
        return ToolResult(
            name="TeamCreate",
            output={"team_name": team_name, "team_file_path": str(team_file), "lead_agent_id": lead_agent_id},
        )


class TeamDeleteTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamDelete",
            description="Disband the current team context.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.team is None:
            return ToolResult(name="TeamDelete", output={"success": False, "message": "No active team"})
        team_name = context.team.get("team_name")
        context.team = None
        team_file = context.workspace_root / ".clawd" / "team.json"
        if team_file.exists():
            try:
                team_file.unlink()
            except Exception:
                pass
        return ToolResult(name="TeamDelete", output={"success": True, "message": "Team deleted", "team_name": team_name})
