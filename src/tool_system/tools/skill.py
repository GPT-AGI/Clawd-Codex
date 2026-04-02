from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolExecutionError
from ..protocol import ToolResult
from ..registry import ToolSpec


class SkillTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Skill",
            description="Execute a user-defined skill from the skills directory.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"name": {"type": "string"}, "input": {"type": "object"}},
                "required": ["name"],
            },
            is_destructive=False,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        name = tool_input["name"]
        if not isinstance(name, str) or not name:
            raise ToolInputError("name must be a non-empty string")
        payload = tool_input.get("input") or {}
        if not isinstance(payload, dict):
            raise ToolInputError("input must be an object when provided")

        skill_dir = Path(os.environ.get("CLAWD_SKILLS_DIR", str(Path.home() / ".clawd" / "skills"))).expanduser().resolve()
        file_path = (skill_dir / f"{name}.py").resolve()
        if not file_path.exists():
            return ToolResult(name="Skill", output={"error": f"skill not found: {name}"}, is_error=True)

        module = _load_module(file_path, module_prefix="clawd_skill_")
        run_fn = getattr(module, "run", None)
        if not callable(run_fn):
            raise ToolExecutionError(f"skill {name} does not export a callable run(input, context)")

        out = run_fn(payload, context)
        return ToolResult(name="Skill", output={"name": name, "output": out})


def _load_module(path: Path, *, module_prefix: str) -> types.ModuleType:
    module_name = f"{module_prefix}{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ToolExecutionError(f"failed to import: {path}")
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, types.ModuleType)
    spec.loader.exec_module(module)
    return module

