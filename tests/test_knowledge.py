"""Tests for knowledge module."""

import os
import tempfile
from pathlib import Path

import pytest

# Import from system-agent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "system-agent"))

from knowledge import (
    get_knowledge,
    get_full_knowledge,
    get_concise_knowledge,
    _get_custom_knowledge_path,
    SYSADMIN_KNOWLEDGE,
    DISTRO_DETECTION,
    EVENT_RESPONSE_PATTERNS,
)


class TestBuiltinKnowledge:
    """Tests for built-in knowledge content."""

    def test_sysadmin_knowledge_has_content(self):
        """Should have substantial sysadmin knowledge."""
        assert len(SYSADMIN_KNOWLEDGE) > 1000
        assert "useradd" in SYSADMIN_KNOWLEDGE
        assert "systemctl" in SYSADMIN_KNOWLEDGE
        assert "journalctl" in SYSADMIN_KNOWLEDGE

    def test_distro_detection_has_content(self):
        """Should have distro detection info."""
        assert "apt" in DISTRO_DETECTION
        assert "dnf" in DISTRO_DETECTION
        assert "pacman" in DISTRO_DETECTION

    def test_event_patterns_has_content(self):
        """Should have event response patterns."""
        assert "High Memory" in EVENT_RESPONSE_PATTERNS
        assert "Low Disk" in EVENT_RESPONSE_PATTERNS
        assert "Service Failed" in EVENT_RESPONSE_PATTERNS


class TestKnowledgeFunctions:
    """Tests for knowledge loading functions."""

    def test_get_full_knowledge(self):
        """Should return combined knowledge."""
        knowledge = get_full_knowledge()
        assert "useradd" in knowledge
        assert "apt" in knowledge
        assert "High Memory" in knowledge

    def test_get_concise_knowledge(self):
        """Should return shorter version."""
        full = get_full_knowledge()
        concise = get_concise_knowledge()
        assert len(concise) < len(full)
        assert "Quick Reference" in concise

    def test_get_knowledge_default(self):
        """Should return full knowledge by default."""
        knowledge = get_knowledge()
        assert "useradd" in knowledge

    def test_get_knowledge_concise(self):
        """Should return concise when requested."""
        knowledge = get_knowledge(concise=True)
        assert "Quick Reference" in knowledge


class TestCustomKnowledge:
    """Tests for custom knowledge loading."""

    def test_custom_path_from_env(self, tmp_path, monkeypatch):
        """Should load from AI_KNOWLEDGE_PATH env var."""
        custom_file = tmp_path / "custom.md"
        custom_file.write_text("# My Custom Knowledge\nCustom content here.")

        monkeypatch.setenv("AI_KNOWLEDGE_PATH", str(custom_file))

        path = _get_custom_knowledge_path()
        assert path == custom_file

        knowledge = get_knowledge()
        assert "My Custom Knowledge" in knowledge
        assert "Custom content here" in knowledge

    def test_custom_path_not_found(self, monkeypatch):
        """Should return None if custom path doesn't exist."""
        monkeypatch.setenv("AI_KNOWLEDGE_PATH", "/nonexistent/path.md")
        # Should fall back to built-in
        knowledge = get_knowledge()
        assert "useradd" in knowledge

    def test_no_custom_returns_builtin(self, monkeypatch):
        """Should return built-in when no custom file."""
        monkeypatch.delenv("AI_KNOWLEDGE_PATH", raising=False)
        knowledge = get_knowledge()
        assert "useradd" in knowledge


class TestKnowledgeContent:
    """Tests for knowledge content quality."""

    def test_has_user_management(self):
        """Should document user management commands."""
        knowledge = get_full_knowledge()
        assert "useradd" in knowledge
        assert "userdel" in knowledge
        assert "usermod" in knowledge

    def test_has_service_management(self):
        """Should document service commands."""
        knowledge = get_full_knowledge()
        assert "systemctl" in knowledge
        assert "journalctl" in knowledge

    def test_has_troubleshooting_heuristics(self):
        """Should include troubleshooting steps."""
        knowledge = get_full_knowledge()
        assert "Service Won't Start" in knowledge
        assert "High CPU" in knowledge
        assert "Disk Full" in knowledge

    def test_has_safety_guidelines(self):
        """Should include safety warnings."""
        knowledge = get_full_knowledge()
        assert "Dangerous Commands" in knowledge
        assert "rm -rf" in knowledge
