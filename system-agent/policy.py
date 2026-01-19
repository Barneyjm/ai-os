"""
AI-OS Agency Policy System

Manages the autonomy level of the AI agent across different domains.
Supports user-defined policies, profiles, and adaptive learning.
"""

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional
import logging
import json
from datetime import datetime

# =============================================================================
# Core Types
# =============================================================================


class AgencyLevel(IntEnum):
    """Levels of AI autonomy, from passive to fully autonomous."""
    OBSERVE = 1      # Watch and log only
    SUGGEST = 2      # Notify user of possible actions
    CONFIRM = 3      # Ask permission before acting
    AUTO = 4         # Act, then notify
    AUTONOMOUS = 5   # Act silently

    @classmethod
    def from_string(cls, s: str) -> "AgencyLevel":
        return cls[s.upper()]

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class Action:
    """Represents an action the AI wants to take."""
    domain: str              # e.g., "filesystem", "processes"
    operation: str           # e.g., "write", "kill"
    target: str              # e.g., "/etc/nginx/nginx.conf", "pid:1234"
    description: str         # Human-readable description
    risk_level: int = 1      # 1-5, used for UI hints
    metadata: dict = field(default_factory=dict)

    @property
    def policy_key(self) -> str:
        """Key used to look up policy for this action."""
        return f"{self.domain}.{self.operation}"


@dataclass
class PolicyDecision:
    """Result of a policy check."""
    level: AgencyLevel
    action: Action
    reason: str              # Why this level was chosen
    source: str              # Where the policy came from (config path)
    requires_confirmation: bool = False
    timeout: int = 30


# =============================================================================
# Policy Engine
# =============================================================================


class AgencyPolicy:
    """
    Manages agency policies and determines appropriate autonomy levels.
    """

    DEFAULT_CONFIG_PATHS = [
        Path("/etc/ai-os/agency.toml"),
        Path.home() / ".config/ai-os/agency.toml",
        Path("config/agency.toml"),  # Local development
    ]

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger("agency-policy")
        self.config: dict = {}
        self.active_profile: Optional[str] = None
        self.response_history: list[dict] = []

        # Load config
        if config_path:
            self._load_config(config_path)
        else:
            for path in self.DEFAULT_CONFIG_PATHS:
                if path.exists():
                    self._load_config(path)
                    break

        if not self.config:
            self.logger.warning("No config found, using defaults")
            self.config = self._default_config()

    def _load_config(self, path: Path):
        """Load configuration from TOML file."""
        try:
            with open(path, "rb") as f:
                self.config = tomllib.load(f)
            self.logger.info(f"Loaded agency policy from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load config from {path}: {e}")

    def _default_config(self) -> dict:
        """Minimal default configuration."""
        return {
            "defaults": {"level": "suggest", "confirmation_timeout": 30},
            "domains": {},
            "profiles": {},
            "learning": {"track_responses": True}
        }

    def get_policy(self, action: Action) -> PolicyDecision:
        """
        Determine the appropriate agency level for an action.

        Resolution order:
        1. Sensitive path overrides
        2. Specific operation policy (e.g., domains.filesystem.write_user_files)
        3. Domain default (e.g., domains.filesystem.level)
        4. Global default (defaults.level)
        """
        domain_config = self.config.get("domains", {}).get(action.domain, {})

        # Check sensitive paths
        if self._is_sensitive_path(action, domain_config):
            return PolicyDecision(
                level=AgencyLevel.CONFIRM,
                action=action,
                reason="Target is a sensitive path",
                source=f"domains.{action.domain}.sensitive_paths",
                requires_confirmation=True,
                timeout=self.config.get("defaults", {}).get("confirmation_timeout", 30)
            )

        # Check protected processes
        if action.domain == "processes" and self._is_protected_process(action, domain_config):
            return PolicyDecision(
                level=AgencyLevel.CONFIRM,
                action=action,
                reason="Target is a protected process",
                source=f"domains.processes.protected_processes",
                requires_confirmation=True,
                timeout=self.config.get("defaults", {}).get("confirmation_timeout", 30)
            )

        # Look up specific operation
        level_str = None
        source = None

        # Try specific operation key
        if action.operation in domain_config:
            level_str = domain_config[action.operation]
            source = f"domains.{action.domain}.{action.operation}"

        # Try domain default
        elif "level" in domain_config:
            level_str = domain_config["level"]
            source = f"domains.{action.domain}.level"

        # Fall back to global default
        else:
            level_str = self.config.get("defaults", {}).get("level", "suggest")
            source = "defaults.level"

        level = AgencyLevel.from_string(level_str)

        return PolicyDecision(
            level=level,
            action=action,
            reason=f"Policy from {source}",
            source=source,
            requires_confirmation=(level == AgencyLevel.CONFIRM),
            timeout=self.config.get("defaults", {}).get("confirmation_timeout", 30)
        )

    def _is_sensitive_path(self, action: Action, domain_config: dict) -> bool:
        """Check if action target is a sensitive path."""
        sensitive = domain_config.get("sensitive_paths", [])
        target = action.target

        for path in sensitive:
            # Expand ~ to home directory
            expanded = str(Path(path).expanduser())
            if target.startswith(expanded) or target == expanded:
                return True

        return False

    def _is_protected_process(self, action: Action, domain_config: dict) -> bool:
        """Check if action target is a protected process."""
        protected = domain_config.get("protected_processes", [])

        # Extract process name from target
        target = action.target
        if ":" in target:
            target = action.metadata.get("process_name", target)

        return target in protected

    # =========================================================================
    # Profiles
    # =========================================================================

    def set_profile(self, profile_name: str) -> bool:
        """Activate a named profile, overlaying its settings on current config."""
        profiles = self.config.get("profiles", {})

        if profile_name not in profiles:
            self.logger.error(f"Unknown profile: {profile_name}")
            return False

        profile = profiles[profile_name]
        self.active_profile = profile_name
        self.logger.info(f"Activated profile: {profile_name}")

        # Apply profile overrides
        # This is a shallow merge - profile settings override base settings
        self._apply_profile_overrides(profile)

        return True

    def _apply_profile_overrides(self, profile: dict):
        """Apply profile overrides to current config."""
        for key, value in profile.items():
            if key in ("description",):
                continue

            # Handle nested keys like "defaults.level"
            if "." in key:
                parts = key.split(".")
                target = self.config
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value
            else:
                self.config[key] = value

    def list_profiles(self) -> dict[str, str]:
        """Return available profiles with descriptions."""
        profiles = self.config.get("profiles", {})
        return {
            name: profile.get("description", "No description")
            for name, profile in profiles.items()
        }

    def get_active_profile(self) -> Optional[str]:
        """Return the currently active profile name."""
        return self.active_profile

    # =========================================================================
    # Learning & Adaptation
    # =========================================================================

    def record_response(self, action: Action, decision: PolicyDecision,
                        user_approved: bool, response_time_ms: int):
        """Record user response for learning."""
        if not self.config.get("learning", {}).get("track_responses", True):
            return

        record = {
            "timestamp": datetime.now().isoformat(),
            "action": {
                "domain": action.domain,
                "operation": action.operation,
                "target": action.target,
            },
            "decision_level": str(decision.level),
            "decision_source": decision.source,
            "user_approved": user_approved,
            "response_time_ms": response_time_ms,
        }

        self.response_history.append(record)
        self._check_for_suggestions()

    def _check_for_suggestions(self):
        """Analyze response history and generate policy suggestions."""
        learning_config = self.config.get("learning", {})
        approval_threshold = learning_config.get("approval_threshold", 5)
        denial_threshold = learning_config.get("denial_threshold", 3)
        no_auto_learn = learning_config.get("no_auto_learn", [])

        # Group recent responses by action type
        from collections import defaultdict
        action_responses = defaultdict(list)

        for record in self.response_history[-100:]:  # Last 100 responses
            key = f"{record['action']['domain']}.{record['action']['operation']}"
            action_responses[key].append(record["user_approved"])

        suggestions = []

        for action_key, responses in action_responses.items():
            # Skip actions that shouldn't auto-learn
            if any(action_key.startswith(pattern.replace("*", ""))
                   for pattern in no_auto_learn):
                continue

            recent = responses[-10:]  # Last 10 of this action type

            if len(recent) >= approval_threshold and all(recent[-approval_threshold:]):
                suggestions.append({
                    "action": action_key,
                    "suggestion": "upgrade",
                    "reason": f"Approved {approval_threshold} times in a row"
                })

            elif len(recent) >= denial_threshold and not any(recent[-denial_threshold:]):
                suggestions.append({
                    "action": action_key,
                    "suggestion": "downgrade",
                    "reason": f"Denied {denial_threshold} times in a row"
                })

        if suggestions:
            self.pending_suggestions = suggestions

    def get_pending_suggestions(self) -> list[dict]:
        """Return any pending policy adjustment suggestions."""
        return getattr(self, "pending_suggestions", [])

    def clear_suggestions(self):
        """Clear pending suggestions."""
        self.pending_suggestions = []


# =============================================================================
# CLI for Policy Management
# =============================================================================


def main():
    """CLI for viewing and managing agency policy."""
    import argparse

    parser = argparse.ArgumentParser(description="AI-OS Agency Policy Manager")
    subparsers = parser.add_subparsers(dest="command")

    # Show current policy
    show_parser = subparsers.add_parser("show", help="Show current policy")
    show_parser.add_argument("--domain", help="Filter by domain")

    # List profiles
    subparsers.add_parser("profiles", help="List available profiles")

    # Set profile
    set_parser = subparsers.add_parser("set-profile", help="Activate a profile")
    set_parser.add_argument("profile", help="Profile name")

    # Check action
    check_parser = subparsers.add_parser("check", help="Check policy for an action")
    check_parser.add_argument("domain", help="Domain (e.g., filesystem)")
    check_parser.add_argument("operation", help="Operation (e.g., write)")
    check_parser.add_argument("target", help="Target (e.g., /etc/nginx.conf)")

    args = parser.parse_args()
    policy = AgencyPolicy()

    if args.command == "show":
        import pprint
        if args.domain:
            pprint.pprint(policy.config.get("domains", {}).get(args.domain, {}))
        else:
            pprint.pprint(policy.config)

    elif args.command == "profiles":
        for name, desc in policy.list_profiles().items():
            print(f"  {name:20} {desc}")

    elif args.command == "set-profile":
        if policy.set_profile(args.profile):
            print(f"Activated profile: {args.profile}")
        else:
            print(f"Failed to activate profile: {args.profile}")

    elif args.command == "check":
        action = Action(
            domain=args.domain,
            operation=args.operation,
            target=args.target,
            description=f"{args.operation} {args.target}"
        )
        decision = policy.get_policy(action)
        print(f"Action: {action.policy_key}")
        print(f"Target: {action.target}")
        print(f"Level:  {decision.level}")
        print(f"Source: {decision.source}")
        print(f"Reason: {decision.reason}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
