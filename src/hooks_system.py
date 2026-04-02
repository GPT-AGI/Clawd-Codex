"""
Hook system for pre/post tool execution.

Provides functionality to:
- Register hooks for specific events
- Execute hooks before/after tool use
- Support command, prompt, and agent hooks
- Handle hook errors gracefully
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class HookResult:
    """Result of hook execution."""

    success: bool
    output: str | None = None
    error: str | None = None
    should_block: bool = False
    reason: str | None = None


class Hook:
    """Base class for hooks."""

    def __init__(self, name: str):
        self.name = name

    def execute(self, context: dict[str, Any]) -> HookResult:
        """Execute the hook."""
        raise NotImplementedError


class CommandHook(Hook):
    """Hook that executes a shell command."""

    def __init__(self, name: str, command: str, timeout: int = 30):
        super().__init__(name)
        self.command = command
        self.timeout = timeout

    def execute(self, context: dict[str, Any]) -> HookResult:
        """Execute shell command."""
        try:
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0:
                return HookResult(success=True, output=result.stdout)
            else:
                return HookResult(
                    success=False, error=result.stderr, should_block=False
                )

        except subprocess.TimeoutExpired:
            return HookResult(success=False, error="Hook timed out")
        except Exception as e:
            return HookResult(success=False, error=str(e))


class PromptHook(Hook):
    """Hook that evaluates a prompt with LLM."""

    def __init__(self, name: str, prompt: str):
        super().__init__(name)
        self.prompt = prompt

    def execute(self, context: dict[str, Any]) -> HookResult:
        """Evaluate prompt (simplified implementation)."""
        # In a full implementation, this would call an LLM
        # For now, return success
        return HookResult(success=True, output=f"Evaluated: {self.prompt[:50]}")


class HookRegistry:
    """Registry for hooks."""

    def __init__(self):
        self.pre_tool_hooks: dict[str, list[Hook]] = {}
        self.post_tool_hooks: dict[str, list[Hook]] = {}

    def register_pre_tool_hook(self, tool_name: str, hook: Hook) -> None:
        """Register a pre-tool hook."""
        if tool_name not in self.pre_tool_hooks:
            self.pre_tool_hooks[tool_name] = []
        self.pre_tool_hooks[tool_name].append(hook)

    def register_post_tool_hook(self, tool_name: str, hook: Hook) -> None:
        """Register a post-tool hook."""
        if tool_name not in self.post_tool_hooks:
            self.post_tool_hooks[tool_name] = []
        self.post_tool_hooks[tool_name].append(hook)

    def get_pre_tool_hooks(self, tool_name: str) -> list[Hook]:
        """Get pre-tool hooks for a tool."""
        return self.pre_tool_hooks.get(tool_name, [])

    def get_post_tool_hooks(self, tool_name: str) -> list[Hook]:
        """Get post-tool hooks for a tool."""
        return self.post_tool_hooks.get(tool_name, [])


class HookExecutor:
    """Executes hooks for tool operations."""

    def __init__(self, registry: HookRegistry):
        self.registry = registry

    def execute_pre_tool_hooks(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[bool, list[HookResult]]:
        """
        Execute pre-tool hooks.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters

        Returns:
            Tuple of (should_proceed, hook_results)
        """
        hooks = self.registry.get_pre_tool_hooks(tool_name)
        results = []

        for hook in hooks:
            context = {"tool_name": tool_name, "tool_input": tool_input}
            result = hook.execute(context)
            results.append(result)

            if result.should_block:
                return False, results

        return True, results

    def execute_post_tool_hooks(
        self, tool_name: str, tool_input: dict[str, Any], tool_output: Any
    ) -> list[HookResult]:
        """
        Execute post-tool hooks.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters
            tool_output: Tool output

        Returns:
            List of hook results
        """
        hooks = self.registry.get_post_tool_hooks(tool_name)
        results = []

        for hook in hooks:
            context = {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output,
            }
            result = hook.execute(context)
            results.append(result)

        return results


def create_hook_registry() -> HookRegistry:
    """Create a hook registry."""
    return HookRegistry()


def create_hook_executor(registry: HookRegistry | None = None) -> HookExecutor:
    """
    Create a hook executor.

    Args:
        registry: Optional hook registry

    Returns:
        HookExecutor instance
    """
    if registry is None:
        registry = create_hook_registry()
    return HookExecutor(registry)
