"""
Tests for the System Agent module.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from agent import (
    AgentConfig,
    Tool,
    ToolRegistry,
    InferenceClient,
    AnthropicInferenceClient,
    create_inference_client,
    SystemAgent,
)
from policy import Action


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = AgentConfig()
        assert config.runtime_socket == "/run/ai-runtime.sock"
        assert config.agent_socket == "/run/system-agent.sock"
        assert config.max_context == 8192
        assert config.inference_backend == "openai"

    def test_from_env(self):
        """Should load config from environment."""
        with patch.dict(
            "os.environ",
            {
                "AI_RUNTIME_URL": "http://test:8080",
                "AI_MODEL_NAME": "test-model",
                "AI_INFERENCE_BACKEND": "openai",
            },
        ):
            config = AgentConfig.from_env()
            assert config.runtime_url == "http://test:8080"
            assert config.model_name == "test-model"
            assert config.inference_backend == "openai"

    def test_from_env_anthropic_autodetect(self):
        """Should auto-detect Anthropic backend when API key is set."""
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "test-key",
            },
            clear=True,
        ):
            config = AgentConfig.from_env()
            assert config.inference_backend == "anthropic"
            assert config.anthropic_api_key == "test-key"

    def test_from_env_openai_api_key(self):
        """Should load OpenAI API key from environment."""
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "sk-test-key",
                "AI_RUNTIME_URL": "https://api.fireworks.ai/inference/v1",
            },
            clear=True,
        ):
            config = AgentConfig.from_env()
            assert config.openai_api_key == "sk-test-key"
            assert config.inference_backend == "openai"

    def test_from_env_explicit_backend_override(self):
        """Should use explicit backend setting over auto-detect."""
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "anthropic-key",
                "AI_INFERENCE_BACKEND": "openai",
            },
            clear=True,
        ):
            config = AgentConfig.from_env()
            # Explicit setting should override auto-detection
            assert config.inference_backend == "openai"


class TestTool:
    """Tests for Tool dataclass."""

    def test_to_action(self):
        """Should convert tool call to Action."""

        async def dummy_handler(**kwargs):
            return {"success": True}

        tool = Tool(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            handler=dummy_handler,
            domain="filesystem",
            operation="read",
        )

        action = tool.to_action(path="/etc/passwd")
        assert action.domain == "filesystem"
        assert action.operation == "read"
        assert action.target == "/etc/passwd"
        assert action.metadata["path"] == "/etc/passwd"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_builtin_tools_registered(self):
        """Should register all builtin tools."""
        registry = ToolRegistry()

        expected_tools = [
            "read_file",
            "write_file",
            "list_directory",
            "find_files",
            "run_command",
            "list_processes",
            "kill_process",
            "service_status",
            "service_control",
            "package_search",
            "package_install",
            "system_info",
            "system_power",
        ]

        for tool_name in expected_tools:
            assert tool_name in registry.tools, f"Missing tool: {tool_name}"

    def test_get_tool_schemas_openai_format(self):
        """Should return OpenAI-compatible tool schemas."""
        registry = ToolRegistry()
        schemas = registry.get_tool_schemas(format="openai")

        assert len(schemas) > 0

        # Check structure of first schema
        schema = schemas[0]
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_get_tool_schemas_anthropic_format(self):
        """Should return Anthropic-compatible tool schemas."""
        registry = ToolRegistry()
        schemas = registry.get_tool_schemas(format="anthropic")

        assert len(schemas) > 0

        # Check structure of first schema
        schema = schemas[0]
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema
        # Anthropic format should NOT have nested "function"
        assert "function" not in schema

    def test_register_custom_tool(self):
        """Should allow registering custom tools."""
        registry = ToolRegistry()

        async def custom_handler(**kwargs):
            return {"result": "custom"}

        tool = Tool(
            name="custom_tool",
            description="A custom tool",
            parameters={"type": "object", "properties": {}},
            handler=custom_handler,
            domain="custom",
            operation="custom_op",
        )

        registry.register(tool)
        assert "custom_tool" in registry.tools


class TestToolImplementations:
    """Tests for tool handler implementations."""

    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    @pytest.mark.asyncio
    async def test_read_file_success(self, registry, tmp_path):
        """Should read file contents."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = await registry._read_file(str(test_file))
        assert result["success"] is True
        assert result["content"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, registry):
        """Should return error for missing file."""
        result = await registry._read_file("/nonexistent/file.txt")
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_write_file_success(self, registry, tmp_path):
        """Should write file contents."""
        test_file = tmp_path / "output.txt"

        result = await registry._write_file(str(test_file), "Test content")
        assert result["success"] is True
        assert test_file.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(self, registry, tmp_path):
        """Should create parent directories."""
        test_file = tmp_path / "nested" / "dir" / "file.txt"

        result = await registry._write_file(str(test_file), "Nested content")
        assert result["success"] is True
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_list_directory(self, registry, tmp_path):
        """Should list directory contents."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "subdir").mkdir()

        result = await registry._list_directory(str(tmp_path))
        assert result["success"] is True
        assert len(result["items"]) == 3

    @pytest.mark.asyncio
    async def test_list_directory_recursive(self, registry, tmp_path):
        """Should list directory recursively."""
        (tmp_path / "file1.txt").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").touch()

        result = await registry._list_directory(str(tmp_path), recursive=True)
        assert result["success"] is True
        # Should include subdir, file1.txt, and subdir/nested.txt
        assert len(result["items"]) >= 3

    @pytest.mark.asyncio
    async def test_find_files(self, registry, tmp_path):
        """Should find files matching pattern."""
        (tmp_path / "test1.py").touch()
        (tmp_path / "test2.py").touch()
        (tmp_path / "other.txt").touch()

        result = await registry._find_files(str(tmp_path), "*.py")
        assert result["success"] is True
        assert len(result["matches"]) == 2

    @pytest.mark.asyncio
    async def test_run_command_success(self, registry):
        """Should run shell command."""
        result = await registry._run_command("echo 'hello'")
        assert result["success"] is True
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_run_command_failure(self, registry):
        """Should handle command failure."""
        result = await registry._run_command("exit 1")
        assert result["success"] is False
        assert result["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_run_command_timeout(self, registry):
        """Should handle command timeout."""
        result = await registry._run_command("sleep 10", timeout=1)
        assert result["success"] is False
        assert "timed out" in result["error"].lower()


class TestInferenceClient:
    """Tests for OpenAI-compatible InferenceClient."""

    def test_init_http_mode(self, mock_agent_config):
        """Should detect HTTP mode from runtime_url."""
        client = InferenceClient(mock_agent_config)
        assert client.use_http is True

    def test_init_socket_mode(self):
        """Should use socket mode when no runtime_url."""
        config = AgentConfig(runtime_url="")
        client = InferenceClient(config)
        assert client.use_http is False

    def test_get_headers_no_api_key(self):
        """Should return minimal headers when no API key."""
        config = AgentConfig(runtime_url="http://localhost:8080")
        client = InferenceClient(config)
        headers = client._get_headers()
        assert headers == {"Content-Type": "application/json"}
        assert "Authorization" not in headers

    def test_get_headers_with_api_key(self):
        """Should include Authorization header when API key is set."""
        config = AgentConfig(
            runtime_url="https://api.fireworks.ai/inference/v1",
            openai_api_key="fw-test-key"
        )
        client = InferenceClient(config)
        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer fw-test-key"


class TestAnthropicInferenceClient:
    """Tests for Anthropic inference client."""

    def test_init(self, mock_anthropic_config):
        """Should initialize with config."""
        client = AnthropicInferenceClient(mock_anthropic_config)
        assert client.config.model_name == "claude-sonnet-4-20250514"

    def test_convert_messages_simple(self, mock_anthropic_config):
        """Should convert simple messages to Anthropic format."""
        client = AnthropicInferenceClient(mock_anthropic_config)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        anthropic_msgs, system = client._convert_messages_to_anthropic(messages)

        assert system == "You are helpful."
        assert len(anthropic_msgs) == 2
        assert anthropic_msgs[0]["role"] == "user"
        assert anthropic_msgs[1]["role"] == "assistant"

    def test_convert_messages_with_tool_calls(self, mock_anthropic_config):
        """Should convert messages with tool calls."""
        client = AnthropicInferenceClient(mock_anthropic_config)

        messages = [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": "I'll list the files.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "list_directory",
                            "arguments": '{"path": "/home"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"items": ["file1", "file2"]}',
            },
        ]

        anthropic_msgs, _ = client._convert_messages_to_anthropic(messages)

        # Check assistant message has tool_use block
        assert anthropic_msgs[1]["role"] == "assistant"
        assert isinstance(anthropic_msgs[1]["content"], list)
        assert any(b["type"] == "tool_use" for b in anthropic_msgs[1]["content"])

        # Check tool result is in user message
        assert anthropic_msgs[2]["role"] == "user"
        assert isinstance(anthropic_msgs[2]["content"], list)
        assert anthropic_msgs[2]["content"][0]["type"] == "tool_result"

    def test_convert_response_text_only(self, mock_anthropic_config):
        """Should convert text-only response."""
        client = AnthropicInferenceClient(mock_anthropic_config)

        anthropic_response = {
            "id": "msg_123",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        openai_response = client._convert_response_to_openai(anthropic_response)

        assert openai_response["choices"][0]["message"]["content"] == "Hello!"
        assert openai_response["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in openai_response["choices"][0]["message"]

    def test_convert_response_with_tool_use(self, mock_anthropic_config):
        """Should convert response with tool use."""
        client = AnthropicInferenceClient(mock_anthropic_config)

        anthropic_response = {
            "id": "msg_123",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "I'll check that."},
                {
                    "type": "tool_use",
                    "id": "call_456",
                    "name": "read_file",
                    "input": {"path": "/etc/hosts"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        openai_response = client._convert_response_to_openai(anthropic_response)

        message = openai_response["choices"][0]["message"]
        assert message["content"] == "I'll check that."
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "read_file"
        assert openai_response["choices"][0]["finish_reason"] == "tool_calls"


class TestCreateInferenceClient:
    """Tests for inference client factory."""

    def test_create_openai_client(self, mock_agent_config):
        """Should create OpenAI client for openai backend."""
        mock_agent_config.inference_backend = "openai"
        client = create_inference_client(mock_agent_config)
        assert isinstance(client, InferenceClient)

    def test_create_anthropic_client(self, mock_anthropic_config):
        """Should create Anthropic client for anthropic backend."""
        client = create_inference_client(mock_anthropic_config)
        assert isinstance(client, AnthropicInferenceClient)

    def test_anthropic_requires_api_key(self):
        """Should raise error if Anthropic backend without API key."""
        config = AgentConfig(
            inference_backend="anthropic",
            anthropic_api_key="",
        )
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            create_inference_client(config)


class TestConfirmationFlow:
    """Tests for the confirmation flow in SystemAgent."""

    @pytest.fixture
    def agent_with_policy(self, mock_agent_config, temp_config_file):
        """Create agent with policy config."""
        mock_agent_config.policy_config = str(temp_config_file)
        return SystemAgent(mock_agent_config)

    @pytest.mark.asyncio
    async def test_confirm_callback_returns_none_pauses_execution(self, agent_with_policy):
        """Should return pending_confirmation when callback returns None."""
        from policy import AgencyLevel

        # Create a mock inference response that triggers a tool call
        mock_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I'll delete that file.",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "run_command",
                            "arguments": '{"command": "rm /tmp/test"}'
                        }
                    }]
                }
            }]
        }

        agent_with_policy.inference.complete = AsyncMock(return_value=mock_response)

        # Use a callback that returns None (needs confirmation)
        async def confirm_callback(action, decision):
            return None

        result = await agent_with_policy.process_message(
            "test-session",
            "delete /tmp/test",
            confirm_callback=confirm_callback
        )

        assert result.get("pending_confirmation") is True
        assert "action_id" in result
        assert "action" in result
        assert result["action"]["domain"] == "processes"
        assert result["action"]["operation"] == "run_command"

    @pytest.mark.asyncio
    async def test_resume_after_confirmation_approved(self, agent_with_policy):
        """Should execute action when confirmation is approved."""
        # First, set up a pending confirmation
        agent_with_policy.pending_confirmations["test-session"] = {
            "action_id": "processes.run_command.1234",
            "tool_name": "run_command",
            "tool_args": {"command": "echo hello"},
            "tool_call_id": "call_123",
            "action": {
                "domain": "processes",
                "operation": "run_command",
                "target": "echo hello",
                "description": "run_command: echo hello"
            },
            "decision": {
                "level": "confirm",
                "reason": "Policy requires confirmation"
            }
        }

        # Set up conversation
        agent_with_policy.conversations["test-session"] = [
            {"role": "system", "content": "You are a system agent."},
            {"role": "user", "content": "run echo hello"},
            {
                "role": "assistant",
                "content": "I'll run that command.",
                "tool_calls": [{
                    "id": "call_123",
                    "function": {
                        "name": "run_command",
                        "arguments": '{"command": "echo hello"}'
                    }
                }]
            }
        ]

        # Mock the inference to return a final response
        mock_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Command executed successfully."
                }
            }]
        }
        agent_with_policy.inference.complete = AsyncMock(return_value=mock_response)

        result = await agent_with_policy.resume_after_confirmation("test-session", approved=True)

        # Should have cleared the pending confirmation
        assert "test-session" not in agent_with_policy.pending_confirmations
        # Should have a response
        assert "response" in result

    @pytest.mark.asyncio
    async def test_resume_after_confirmation_denied(self, agent_with_policy):
        """Should not execute action when confirmation is denied."""
        # Set up a pending confirmation
        agent_with_policy.pending_confirmations["test-session"] = {
            "action_id": "processes.run_command.1234",
            "tool_name": "run_command",
            "tool_args": {"command": "rm -rf /"},
            "tool_call_id": "call_123",
            "action": {
                "domain": "processes",
                "operation": "run_command",
                "target": "rm -rf /",
                "description": "run_command: rm -rf /"
            },
            "decision": {
                "level": "confirm",
                "reason": "Policy requires confirmation"
            }
        }

        agent_with_policy.conversations["test-session"] = [
            {"role": "system", "content": "You are a system agent."},
            {"role": "user", "content": "delete everything"},
            {
                "role": "assistant",
                "content": "I'll delete that.",
                "tool_calls": [{
                    "id": "call_123",
                    "function": {
                        "name": "run_command",
                        "arguments": '{"command": "rm -rf /"}'
                    }
                }]
            }
        ]

        # Mock the inference to return a final response
        mock_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Action was denied by user."
                }
            }]
        }
        agent_with_policy.inference.complete = AsyncMock(return_value=mock_response)

        result = await agent_with_policy.resume_after_confirmation("test-session", approved=False)

        # Check that the tool result indicates denial
        messages = agent_with_policy.conversations["test-session"]
        tool_result = json.loads(messages[-1]["content"])
        assert tool_result["policy"] == "denied"

    @pytest.mark.asyncio
    async def test_resume_without_pending_confirmation(self, agent_with_policy):
        """Should return error when no pending confirmation."""
        result = await agent_with_policy.resume_after_confirmation("nonexistent-session", approved=True)
        assert "error" in result

    def test_pending_confirmations_initialized(self, agent_with_policy):
        """Should initialize pending_confirmations dict."""
        assert hasattr(agent_with_policy, "pending_confirmations")
        assert isinstance(agent_with_policy.pending_confirmations, dict)


class TestSystemAgent:
    """Tests for SystemAgent class."""

    def test_init_with_openai_backend(self, mock_agent_config):
        """Should initialize with OpenAI backend."""
        agent = SystemAgent(mock_agent_config)
        assert agent.tool_schema_format == "openai"
        assert isinstance(agent.inference, InferenceClient)

    def test_init_with_anthropic_backend(self, mock_anthropic_config):
        """Should initialize with Anthropic backend."""
        agent = SystemAgent(mock_anthropic_config)
        assert agent.tool_schema_format == "anthropic"
        assert isinstance(agent.inference, AnthropicInferenceClient)

    def test_get_system_prompt(self, mock_agent_config):
        """Should generate system prompt."""
        agent = SystemAgent(mock_agent_config)
        prompt = agent._get_system_prompt()

        assert "System Agent" in prompt
        assert "AI-first operating system" in prompt
        assert "tools" in prompt.lower()

    def test_get_system_prompt_with_profile(self, mock_agent_config):
        """Should include profile info in prompt."""
        agent = SystemAgent(mock_agent_config)
        agent.policy.active_profile = "focus"

        prompt = agent._get_system_prompt()
        assert "focus" in prompt

    def test_set_profile(self, mock_agent_config, temp_config_file):
        """Should change agency profile."""
        mock_agent_config.policy_config = str(temp_config_file)
        agent = SystemAgent(mock_agent_config)

        assert agent.set_profile("cautious") is True
        assert agent.policy.active_profile == "cautious"

    def test_get_policy_summary(self, mock_agent_config, temp_config_file):
        """Should return policy summary."""
        mock_agent_config.policy_config = str(temp_config_file)
        agent = SystemAgent(mock_agent_config)

        summary = agent.get_policy_summary()
        assert "active_profile" in summary
        assert "available_profiles" in summary
        assert "pending_suggestions" in summary
