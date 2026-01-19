"""
Tests for the Agency Policy System.
"""

import pytest
from pathlib import Path

from policy import AgencyLevel, Action, PolicyDecision, AgencyPolicy


class TestAgencyLevel:
    """Tests for AgencyLevel enum."""

    def test_level_ordering(self):
        """Agency levels should be ordered from least to most autonomous."""
        assert AgencyLevel.OBSERVE < AgencyLevel.SUGGEST
        assert AgencyLevel.SUGGEST < AgencyLevel.CONFIRM
        assert AgencyLevel.CONFIRM < AgencyLevel.AUTO
        assert AgencyLevel.AUTO < AgencyLevel.AUTONOMOUS

    def test_from_string(self):
        """Should convert string to AgencyLevel."""
        assert AgencyLevel.from_string("observe") == AgencyLevel.OBSERVE
        assert AgencyLevel.from_string("SUGGEST") == AgencyLevel.SUGGEST
        assert AgencyLevel.from_string("Confirm") == AgencyLevel.CONFIRM
        assert AgencyLevel.from_string("auto") == AgencyLevel.AUTO
        assert AgencyLevel.from_string("autonomous") == AgencyLevel.AUTONOMOUS

    def test_from_string_invalid(self):
        """Should raise KeyError for invalid level string."""
        with pytest.raises(KeyError):
            AgencyLevel.from_string("invalid")

    def test_str_representation(self):
        """String representation should be lowercase."""
        assert str(AgencyLevel.OBSERVE) == "observe"
        assert str(AgencyLevel.AUTONOMOUS) == "autonomous"


class TestAction:
    """Tests for Action dataclass."""

    def test_action_creation(self):
        """Should create action with all fields."""
        action = Action(
            domain="filesystem",
            operation="write",
            target="/etc/nginx.conf",
            description="Write nginx config",
            risk_level=3,
            metadata={"backup": True},
        )
        assert action.domain == "filesystem"
        assert action.operation == "write"
        assert action.target == "/etc/nginx.conf"
        assert action.risk_level == 3
        assert action.metadata["backup"] is True

    def test_policy_key(self):
        """Policy key should combine domain and operation."""
        action = Action(
            domain="filesystem",
            operation="delete_user_files",
            target="/home/user/file.txt",
            description="Delete file",
        )
        assert action.policy_key == "filesystem.delete_user_files"

    def test_default_values(self):
        """Should have sensible defaults."""
        action = Action(
            domain="test",
            operation="test_op",
            target="target",
            description="desc",
        )
        assert action.risk_level == 1
        assert action.metadata == {}


class TestAgencyPolicy:
    """Tests for AgencyPolicy class."""

    def test_default_config(self):
        """Should use default config when no file found."""
        policy = AgencyPolicy(config_path=Path("/nonexistent/path.toml"))
        assert policy.config is not None
        assert "defaults" in policy.config

    def test_load_config(self, temp_config_file):
        """Should load config from file."""
        policy = AgencyPolicy(config_path=temp_config_file)
        assert policy.config["defaults"]["level"] == "suggest"

    def test_get_policy_default_level(self, temp_config_file):
        """Should return default level for unknown operations."""
        policy = AgencyPolicy(config_path=temp_config_file)
        action = Action(
            domain="unknown",
            operation="unknown_op",
            target="target",
            description="Unknown action",
        )
        decision = policy.get_policy(action)
        assert decision.level == AgencyLevel.SUGGEST
        assert decision.source == "defaults.level"

    def test_get_policy_domain_operation(self, temp_config_file):
        """Should return specific operation policy."""
        policy = AgencyPolicy(config_path=temp_config_file)
        action = Action(
            domain="filesystem",
            operation="read",
            target="/home/user/file.txt",
            description="Read file",
        )
        decision = policy.get_policy(action)
        assert decision.level == AgencyLevel.AUTO
        assert decision.source == "domains.filesystem.read"

    def test_get_policy_sensitive_path(self, temp_config_file):
        """Should require confirmation for sensitive paths."""
        policy = AgencyPolicy(config_path=temp_config_file)
        action = Action(
            domain="filesystem",
            operation="write_user_files",
            target="/etc/passwd",
            description="Write to passwd",
        )
        decision = policy.get_policy(action)
        assert decision.level == AgencyLevel.CONFIRM
        assert decision.requires_confirmation is True
        assert "sensitive" in decision.reason.lower()

    def test_get_policy_sensitive_path_home_expansion(self, temp_config_file):
        """Should expand ~ in sensitive paths."""
        policy = AgencyPolicy(config_path=temp_config_file)
        home = str(Path.home())
        action = Action(
            domain="filesystem",
            operation="write_user_files",
            target=f"{home}/.ssh/id_rsa",
            description="Write SSH key",
        )
        decision = policy.get_policy(action)
        assert decision.level == AgencyLevel.CONFIRM
        assert decision.requires_confirmation is True

    def test_get_policy_protected_process(self, temp_config_file):
        """Should require confirmation for protected processes."""
        policy = AgencyPolicy(config_path=temp_config_file)
        action = Action(
            domain="processes",
            operation="kill_user_processes",
            target="init",
            description="Kill init",
            metadata={"process_name": "init"},
        )
        decision = policy.get_policy(action)
        assert decision.level == AgencyLevel.CONFIRM
        assert "protected" in decision.reason.lower()


class TestAgencyPolicyProfiles:
    """Tests for profile management."""

    def test_list_profiles(self, temp_config_file):
        """Should list available profiles."""
        policy = AgencyPolicy(config_path=temp_config_file)
        profiles = policy.list_profiles()
        assert "default" in profiles
        assert "cautious" in profiles
        assert "focus" in profiles

    def test_set_profile(self, temp_config_file):
        """Should activate a profile."""
        policy = AgencyPolicy(config_path=temp_config_file)
        assert policy.set_profile("cautious") is True
        assert policy.active_profile == "cautious"

    def test_set_invalid_profile(self, temp_config_file):
        """Should return False for invalid profile."""
        policy = AgencyPolicy(config_path=temp_config_file)
        assert policy.set_profile("nonexistent") is False
        assert policy.active_profile is None

    def test_get_active_profile(self, temp_config_file):
        """Should return active profile name."""
        policy = AgencyPolicy(config_path=temp_config_file)
        assert policy.get_active_profile() is None
        policy.set_profile("focus")
        assert policy.get_active_profile() == "focus"


class TestAgencyPolicyLearning:
    """Tests for learning and adaptation features."""

    def test_record_response(self, temp_config_file):
        """Should record user responses."""
        policy = AgencyPolicy(config_path=temp_config_file)
        action = Action(
            domain="filesystem",
            operation="write",
            target="/tmp/test",
            description="Test write",
        )
        decision = PolicyDecision(
            level=AgencyLevel.CONFIRM,
            action=action,
            reason="Test",
            source="test",
            requires_confirmation=True,
        )

        policy.record_response(action, decision, user_approved=True, response_time_ms=500)
        assert len(policy.response_history) == 1
        assert policy.response_history[0]["user_approved"] is True

    def test_no_tracking_when_disabled(self, temp_config_file):
        """Should not record when tracking is disabled."""
        policy = AgencyPolicy(config_path=temp_config_file)
        policy.config["learning"]["track_responses"] = False

        action = Action(
            domain="filesystem",
            operation="write",
            target="/tmp/test",
            description="Test write",
        )
        decision = PolicyDecision(
            level=AgencyLevel.CONFIRM,
            action=action,
            reason="Test",
            source="test",
        )

        policy.record_response(action, decision, user_approved=True, response_time_ms=500)
        assert len(policy.response_history) == 0

    def test_get_pending_suggestions_empty(self, temp_config_file):
        """Should return empty list when no suggestions."""
        policy = AgencyPolicy(config_path=temp_config_file)
        assert policy.get_pending_suggestions() == []

    def test_clear_suggestions(self, temp_config_file):
        """Should clear pending suggestions."""
        policy = AgencyPolicy(config_path=temp_config_file)
        policy.pending_suggestions = [{"test": "suggestion"}]
        policy.clear_suggestions()
        assert policy.get_pending_suggestions() == []


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""

    def test_decision_creation(self):
        """Should create decision with all fields."""
        action = Action(
            domain="test",
            operation="op",
            target="t",
            description="d",
        )
        decision = PolicyDecision(
            level=AgencyLevel.CONFIRM,
            action=action,
            reason="Test reason",
            source="test.source",
            requires_confirmation=True,
            timeout=60,
        )
        assert decision.level == AgencyLevel.CONFIRM
        assert decision.requires_confirmation is True
        assert decision.timeout == 60

    def test_decision_defaults(self):
        """Should have sensible defaults."""
        action = Action(
            domain="test",
            operation="op",
            target="t",
            description="d",
        )
        decision = PolicyDecision(
            level=AgencyLevel.AUTO,
            action=action,
            reason="Test",
            source="test",
        )
        assert decision.requires_confirmation is False
        assert decision.timeout == 30
