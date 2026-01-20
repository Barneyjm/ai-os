"""
Shared pytest fixtures for MeerkatOS tests.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add source directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "system-agent"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-shell"))


@pytest.fixture
def sample_agency_config():
    """Return a minimal agency config as a dict."""
    return {
        "defaults": {
            "level": "suggest",
            "confirmation_timeout": 30,
        },
        "domains": {
            "filesystem": {
                "read": "auto",
                "list": "auto",
                "write_user_files": "auto",
                "write_config_files": "confirm",
                "delete_user_files": "confirm",
                "sensitive_paths": [
                    "/etc/passwd",
                    "/etc/shadow",
                    "~/.ssh",
                ],
            },
            "processes": {
                "list": "auto",
                "kill_user_processes": "suggest",
                "kill_system_processes": "confirm",
                "protected_processes": [
                    "init",
                    "systemd",
                    "system-agent",
                ],
            },
            "system": {
                "resources": "auto",
                "shutdown": "confirm",
                "reboot": "confirm",
            },
        },
        "profiles": {
            "default": {
                "description": "Balanced defaults for daily use",
            },
            "cautious": {
                "description": "Maximum confirmation",
                "defaults.level": "confirm",
            },
            "focus": {
                "description": "Minimize interruptions",
                "defaults.level": "auto",
            },
        },
        "learning": {
            "track_responses": True,
            "approval_threshold": 5,
            "denial_threshold": 3,
            "no_auto_learn": ["filesystem.delete", "system.shutdown"],
        },
    }


@pytest.fixture
def temp_config_file(sample_agency_config):
    """Create a temporary config file with sample config."""
    try:
        import tomli_w
    except ImportError:
        # If tomli_w not available, write manually
        import json

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            # Write a simple TOML manually
            f.write('# Test config\n')
            f.write('[defaults]\n')
            f.write('level = "suggest"\n')
            f.write('confirmation_timeout = 30\n\n')
            f.write('[domains.filesystem]\n')
            f.write('read = "auto"\n')
            f.write('list = "auto"\n')
            f.write('write_user_files = "auto"\n')
            f.write('write_config_files = "confirm"\n')
            f.write('delete_user_files = "confirm"\n')
            f.write('sensitive_paths = ["/etc/passwd", "/etc/shadow", "~/.ssh"]\n\n')
            f.write('[domains.processes]\n')
            f.write('list = "auto"\n')
            f.write('kill_user_processes = "suggest"\n')
            f.write('kill_system_processes = "confirm"\n')
            f.write('protected_processes = ["init", "systemd", "system-agent"]\n\n')
            f.write('[domains.system]\n')
            f.write('resources = "auto"\n')
            f.write('shutdown = "confirm"\n')
            f.write('reboot = "confirm"\n\n')
            f.write('[profiles.default]\n')
            f.write('description = "Balanced defaults for daily use"\n\n')
            f.write('[profiles.cautious]\n')
            f.write('description = "Maximum confirmation"\n')
            f.write('"defaults.level" = "confirm"\n\n')
            f.write('[profiles.focus]\n')
            f.write('description = "Minimize interruptions"\n')
            f.write('"defaults.level" = "auto"\n\n')
            f.write('[learning]\n')
            f.write('track_responses = true\n')
            f.write('approval_threshold = 5\n')
            f.write('denial_threshold = 3\n')
            f.write('no_auto_learn = ["filesystem.delete", "system.shutdown"]\n')
            temp_path = f.name

        yield Path(temp_path)
        os.unlink(temp_path)
        return

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".toml", delete=False
    ) as f:
        tomli_w.dump(sample_agency_config, f)
        temp_path = f.name

    yield Path(temp_path)
    os.unlink(temp_path)


@pytest.fixture
def mock_agent_config():
    """Return a mock AgentConfig for testing."""
    from agent import AgentConfig

    return AgentConfig(
        runtime_socket="/tmp/test-runtime.sock",
        runtime_url="http://127.0.0.1:8080",
        agent_socket="/tmp/test-agent.sock",
        model_name="test-model",
        max_context=4096,
        consent_timeout=10,
        log_level="DEBUG",
        policy_config=None,
        anthropic_api_key="test-key",
        anthropic_base_url="https://api.anthropic.com",
        inference_backend="openai",
    )


@pytest.fixture
def mock_anthropic_config():
    """Return a mock AgentConfig configured for Anthropic."""
    from agent import AgentConfig

    return AgentConfig(
        runtime_socket="/tmp/test-runtime.sock",
        runtime_url="",
        agent_socket="/tmp/test-agent.sock",
        model_name="claude-sonnet-4-20250514",
        max_context=8192,
        consent_timeout=30,
        log_level="DEBUG",
        policy_config=None,
        anthropic_api_key="test-anthropic-key",
        anthropic_base_url="https://api.anthropic.com",
        inference_backend="anthropic",
    )
