"""
Tests for the AI Shell module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shell import AgentClient, BUILTIN_COMMANDS


class TestAgentClient:
    """Tests for AgentClient."""

    def test_init_http_mode(self):
        """Should initialize in HTTP mode."""
        client = AgentClient(url="http://127.0.0.1:8000")
        assert client.use_http is True
        assert client.url == "http://127.0.0.1:8000"

    def test_init_socket_mode(self):
        """Should initialize in socket mode."""
        client = AgentClient(socket_path="/run/agent.sock")
        assert client.use_http is False
        assert client.socket_path == "/run/agent.sock"

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Should send message to agent."""
        client = AgentClient(url="http://127.0.0.1:8000")

        mock_response = {
            "response": "Hello!",
            "notifications": [],
            "policy_suggestions": [],
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await client.send_message("Hi")

            mock_request.assert_called_once_with(
                "POST",
                "/chat",
                {"session_id": client.session_id, "message": "Hi"},
            )
            assert result["response"] == "Hello!"

    @pytest.mark.asyncio
    async def test_get_profiles(self):
        """Should get available profiles."""
        client = AgentClient(url="http://127.0.0.1:8000")

        mock_response = {
            "profiles": {"default": "Default profile", "focus": "Focus mode"},
            "active": "default",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await client.get_profiles()

            mock_request.assert_called_once_with("GET", "/profiles")
            assert "default" in result["profiles"]
            assert result["active"] == "default"

    @pytest.mark.asyncio
    async def test_set_profile(self):
        """Should set active profile."""
        client = AgentClient(url="http://127.0.0.1:8000")

        mock_response = {"success": True, "active": "focus"}

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await client.set_profile("focus")

            mock_request.assert_called_once_with(
                "POST", "/profiles/set", {"profile": "focus"}
            )
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_check_policy(self):
        """Should check policy for an action."""
        client = AgentClient(url="http://127.0.0.1:8000")

        mock_response = {
            "level": "confirm",
            "reason": "Sensitive path",
            "source": "domains.filesystem.sensitive_paths",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await client.check_policy(
                "filesystem", "write", "/etc/passwd"
            )

            assert result["level"] == "confirm"

    @pytest.mark.asyncio
    async def test_get_health(self):
        """Should get agent health status."""
        client = AgentClient(url="http://127.0.0.1:8000")

        mock_response = {"status": "healthy"}

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await client.get_health()

            mock_request.assert_called_once_with("GET", "/health")
            assert result["status"] == "healthy"


class TestBuiltinCommands:
    """Tests for shell builtin commands."""

    def test_builtin_commands_defined(self):
        """Should have all expected builtin commands."""
        expected = ["/help", "/clear", "/reset", "/profile", "/policy", "/exit"]
        for cmd in expected:
            assert cmd in BUILTIN_COMMANDS

    def test_builtin_commands_have_descriptions(self):
        """All builtin commands should have descriptions."""
        for cmd, desc in BUILTIN_COMMANDS.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestHandleBuiltin:
    """Tests for handle_builtin function."""

    @pytest.mark.asyncio
    async def test_help_command(self):
        """Should handle /help command."""
        from shell import handle_builtin

        client = MagicMock()
        result = await handle_builtin("/help", client)
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_command(self):
        """Should handle /clear command."""
        from shell import handle_builtin

        client = MagicMock()
        result = await handle_builtin("/clear", client)
        assert result is True

    @pytest.mark.asyncio
    async def test_reset_command(self):
        """Should handle /reset command."""
        from shell import handle_builtin

        client = MagicMock()
        result = await handle_builtin("/reset", client)
        assert result is True

    @pytest.mark.asyncio
    async def test_profile_list_command(self):
        """Should handle /profile list command."""
        from shell import handle_builtin

        client = MagicMock()
        client.get_profiles = AsyncMock(
            return_value={
                "profiles": {"default": "Default", "focus": "Focus"},
                "active": "default",
            }
        )

        result = await handle_builtin("/profile list", client)
        assert result is True
        client.get_profiles.assert_called_once()

    @pytest.mark.asyncio
    async def test_profile_set_command(self):
        """Should handle /profile set command."""
        from shell import handle_builtin

        client = MagicMock()
        client.set_profile = AsyncMock(return_value={"success": True, "active": "focus"})

        result = await handle_builtin("/profile set focus", client)
        assert result is True
        client.set_profile.assert_called_once_with("focus")

    @pytest.mark.asyncio
    async def test_policy_check_command(self):
        """Should handle /policy command."""
        from shell import handle_builtin

        client = MagicMock()
        client.check_policy = AsyncMock(
            return_value={
                "level": "auto",
                "reason": "Policy from config",
                "source": "domains.filesystem.read",
            }
        )

        result = await handle_builtin("/policy filesystem read /tmp/file", client)
        assert result is True
        client.check_policy.assert_called_once_with("filesystem", "read", "/tmp/file")

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        """Should return False for non-builtin commands."""
        from shell import handle_builtin

        client = MagicMock()
        result = await handle_builtin("/unknown", client)
        assert result is False

    @pytest.mark.asyncio
    async def test_exit_command(self):
        """Should handle /exit command by raising SystemExit."""
        from shell import handle_builtin

        client = MagicMock()
        with pytest.raises(SystemExit):
            await handle_builtin("/exit", client)


class TestHelperFunctions:
    """Tests for shell helper functions."""

    def test_get_prompt(self):
        """Should generate prompt with current directory."""
        from shell import get_prompt
        import os

        prompt = get_prompt()
        # Should be an HTML formatted prompt
        assert prompt is not None

    def test_print_response_empty(self, capsys):
        """Should handle empty response."""
        from shell import print_response

        print_response({"response": "", "notifications": []})
        # Should not crash

    def test_print_response_with_notifications(self, capsys):
        """Should print notifications."""
        from shell import print_response

        response = {
            "response": "Done",
            "notifications": [{"message": "File written"}],
            "policy_suggestions": [],
        }
        print_response(response)
        # Should not crash

    def test_print_response_with_markdown(self, capsys):
        """Should handle markdown in response."""
        from shell import print_response

        response = {
            "response": "# Header\n\n```python\nprint('hello')\n```",
            "notifications": [],
            "policy_suggestions": [],
        }
        print_response(response)
        # Should not crash
