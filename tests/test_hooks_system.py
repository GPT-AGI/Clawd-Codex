"""Tests for hooks_system module."""

import unittest

from src.hooks_system import (
    Hook,
    HookResult,
    CommandHook,
    PromptHook,
    HookRegistry,
    HookExecutor,
    create_hook_registry,
    create_hook_executor,
)


class TestHooksSystem(unittest.TestCase):
    """Test cases for hook system."""

    def test_create_hook_result(self):
        """Test creating a hook result."""
        result = HookResult(success=True, output="test")

        self.assertTrue(result.success)
        self.assertEqual(result.output, "test")

    def test_create_command_hook(self):
        """Test creating a command hook."""
        hook = CommandHook("test", "echo 'hello'")

        self.assertEqual(hook.name, "test")
        self.assertEqual(hook.command, "echo 'hello'")

    def test_execute_command_hook_success(self):
        """Test successful command execution."""
        hook = CommandHook("test", "echo 'success'", timeout=5)
        result = hook.execute({})

        self.assertTrue(result.success)
        self.assertIn("success", result.output)

    def test_execute_command_hook_failure(self):
        """Test failed command execution."""
        hook = CommandHook("test", "exit 1", timeout=5)
        result = hook.execute({})

        self.assertFalse(result.success)

    def test_execute_command_hook_timeout(self):
        """Test command timeout."""
        hook = CommandHook("test", "sleep 10", timeout=1)
        result = hook.execute({})

        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)

    def test_create_prompt_hook(self):
        """Test creating a prompt hook."""
        hook = PromptHook("test", "Is this safe?")

        self.assertEqual(hook.name, "test")
        self.assertEqual(hook.prompt, "Is this safe?")

    def test_execute_prompt_hook(self):
        """Test prompt hook execution."""
        hook = PromptHook("test", "Test prompt")
        result = hook.execute({})

        self.assertTrue(result.success)

    def test_create_hook_registry(self):
        """Test creating a hook registry."""
        registry = create_hook_registry()

        self.assertIsInstance(registry, HookRegistry)
        self.assertEqual(len(registry.pre_tool_hooks), 0)

    def test_register_pre_tool_hook(self):
        """Test registering pre-tool hook."""
        registry = create_hook_registry()
        hook = CommandHook("test", "echo 'test'")

        registry.register_pre_tool_hook("Write", hook)

        self.assertIn("Write", registry.pre_tool_hooks)
        self.assertEqual(len(registry.pre_tool_hooks["Write"]), 1)

    def test_register_post_tool_hook(self):
        """Test registering post-tool hook."""
        registry = create_hook_registry()
        hook = CommandHook("test", "echo 'test'")

        registry.register_post_tool_hook("Read", hook)

        self.assertIn("Read", registry.post_tool_hooks)
        self.assertEqual(len(registry.post_tool_hooks["Read"]), 1)

    def test_get_pre_tool_hooks(self):
        """Test getting pre-tool hooks."""
        registry = create_hook_registry()
        hook = CommandHook("test", "echo 'test'")
        registry.register_pre_tool_hook("Write", hook)

        hooks = registry.get_pre_tool_hooks("Write")

        self.assertEqual(len(hooks), 1)
        self.assertEqual(hooks[0].name, "test")

    def test_get_post_tool_hooks(self):
        """Test getting post-tool hooks."""
        registry = create_hook_registry()
        hook = CommandHook("test", "echo 'test'")
        registry.register_post_tool_hook("Read", hook)

        hooks = registry.get_post_tool_hooks("Read")

        self.assertEqual(len(hooks), 1)
        self.assertEqual(hooks[0].name, "test")

    def test_hook_executor_pre_tool(self):
        """Test hook executor for pre-tool hooks."""
        registry = create_hook_registry()
        hook = CommandHook("test", "echo 'pre-tool'")
        registry.register_pre_tool_hook("Write", hook)

        executor = create_hook_executor(registry)
        should_proceed, results = executor.execute_pre_tool_hooks(
            "Write", {"file_path": "/tmp/test"}
        )

        self.assertTrue(should_proceed)
        self.assertEqual(len(results), 1)

    def test_hook_executor_post_tool(self):
        """Test hook executor for post-tool hooks."""
        registry = create_hook_registry()
        hook = CommandHook("test", "echo 'post-tool'")
        registry.register_post_tool_hook("Read", hook)

        executor = create_hook_executor(registry)
        results = executor.execute_post_tool_hooks(
            "Read", {"file_path": "/tmp/test"}, {"content": "test"}
        )

        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
