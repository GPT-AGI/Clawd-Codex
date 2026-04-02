"""Test agent loop with mocked provider to verify tool invocation."""

import unittest
from unittest.mock import MagicMock
from pathlib import Path
import tempfile

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system.defaults import build_default_registry
from src.tool_system.context import ToolContext
from src.tool_system.agent_loop import run_agent_loop


class TestAgentLoop(unittest.TestCase):
    """Test agent loop logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_agent_loop_calls_tool(self):
        """Test agent loop correctly dispatches a tool call from mocked LLM."""
        conversation = Conversation()
        conversation.add_user_message("Create a file hello.py with content print('hello world')")

        # Mock provider
        mock_provider = MagicMock()

        # First response: tool use Write
        mock_tool_use = {
            "id": "toolu_123",
            "name": "Write",
            "input": {
                "file_path": str(self.workspace / "hello.py"),
                "content": "print('hello world')"
            }
        }
        mock_response1 = ChatResponse(
            content="I will create the file.",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            finish_reason="tool_use",
            tool_uses=[mock_tool_use],
        )

        # Second response: final text after tool result
        mock_response2 = ChatResponse(
            content="File created successfully!",
            model="test-model",
            usage={"input_tokens": 30, "output_tokens": 10},
            finish_reason="stop",
            tool_uses=None,
        )

        mock_provider.chat.side_effect = [mock_response1, mock_response2]

        final_text = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )

        # Verify final response
        self.assertEqual(final_text, "File created successfully!")

        # Verify provider was called twice
        self.assertEqual(mock_provider.chat.call_count, 2)

        # Verify file was created
        hello_py = self.workspace / "hello.py"
        self.assertTrue(hello_py.exists())
        self.assertEqual(hello_py.read_text(), "print('hello world')")

    def test_agent_loop_creates_hello_world(self):
        """Test agent loop creates hello.py and writes print('hello world')."""
        conversation = Conversation()
        conversation.add_user_message("Create a file hello.py with content print('hello world')")

        mock_provider = MagicMock()

        # First response: tool use Write
        hello_path = self.workspace / "hello.py"
        mock_tool_write = {
            "id": "toolu_123",
            "name": "Write",
            "input": {
                "file_path": str(hello_path),
                "content": "print('hello world')"
            }
        }
        mock_response1 = ChatResponse(
            content="I will create the file.",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            finish_reason="tool_use",
            tool_uses=[mock_tool_write],
        )

        # Second response: final
        mock_response2 = ChatResponse(
            content="File created successfully!",
            model="test-model",
            usage={"input_tokens": 30, "output_tokens": 10},
            finish_reason="stop",
            tool_uses=None,
        )

        mock_provider.chat.side_effect = [mock_response1, mock_response2]

        final_text = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )

        self.assertEqual(final_text, "File created successfully!")
        self.assertTrue(hello_path.exists())
        self.assertEqual(hello_path.read_text(), "print('hello world')")


if __name__ == "__main__":
    unittest.main()
