from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from typing import Any, Iterable

from .protocol import ToolResult
from .registry import Tool, ToolRegistry, ToolSpec


def load_tools_from_dir(directory: str | Path) -> list[Tool]:
    path = Path(directory).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        return []

    tools: list[Tool] = []
    for file_path in sorted(path.glob("*.py")):
        tool = _load_tool_from_file(file_path)
        if tool is not None:
            tools.append(tool)
    return tools


def _load_tool_from_file(file_path: Path) -> Tool | None:
    module_name = f"clawd_user_tool_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, types.ModuleType)
    spec.loader.exec_module(module)

    tool_obj = getattr(module, "TOOL", None)
    if tool_obj is not None:
        return tool_obj

    tool_spec = getattr(module, "tool_spec", None)
    run_fn = getattr(module, "run", None)
    if not isinstance(tool_spec, dict) or not callable(run_fn):
        return None

    class _FunctionTool:
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name=str(tool_spec["name"]),
                description=str(tool_spec.get("description", "")),
                input_schema=dict(tool_spec.get("input_schema") or {"type": "object"}),
                aliases=tuple(tool_spec.get("aliases") or ()),
                is_read_only=bool(tool_spec.get("is_read_only", False)),
                is_destructive=bool(tool_spec.get("is_destructive", False)),
                strict=bool(tool_spec.get("strict", False)),
                max_result_size_chars=int(tool_spec.get("max_result_size_chars", 20_000)),
            )

        def run(self, tool_input: dict[str, Any], context) -> ToolResult:
            out = run_fn(tool_input, context)
            return out if isinstance(out, ToolResult) else ToolResult(name=self.spec().name, output=out)

    return _FunctionTool()


def load_into_registry(registry: ToolRegistry, directories: Iterable[str | Path]) -> None:
    for directory in directories:
        for tool in load_tools_from_dir(directory):
            registry.register(tool)
