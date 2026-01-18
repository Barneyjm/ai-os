"""
Audit Logging for AI-OS

Provides structured audit logging for all agent actions, policy decisions,
and events. Logs are stored in JSONL format for easy parsing and analysis.
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import fcntl


class AuditAction(Enum):
    """Types of auditable actions."""
    TOOL_INVOKED = "tool_invoked"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    POLICY_CHECK = "policy_check"
    POLICY_DENIED = "policy_denied"
    POLICY_APPROVED = "policy_approved"
    USER_CONFIRMED = "user_confirmed"
    USER_DENIED = "user_denied"
    EVENT_TRIGGERED = "event_triggered"
    EVENT_HANDLED = "event_handled"
    PROFILE_CHANGED = "profile_changed"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: str
    action: str
    session_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[dict] = None
    domain: Optional[str] = None
    operation: Optional[str] = None
    target: Optional[str] = None
    policy_level: Optional[str] = None
    policy_source: Optional[str] = None
    event_type: Optional[str] = None
    event_source: Optional[str] = None
    trigger_id: Optional[str] = None
    profile: Optional[str] = None
    duration_ms: Optional[int] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and value != {}:
                result[key] = value
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "AuditEntry":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class AuditLogger:
    """
    Structured audit logger for AI-OS.

    Writes JSONL format logs to a file with automatic rotation.
    Thread-safe using file locking.
    """

    DEFAULT_LOG_DIR = Path.home() / ".ai-os" / "audit"
    DEFAULT_LOG_FILE = "audit.jsonl"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB before rotation
    MAX_LOG_FILES = 5  # Keep 5 rotated files

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_file: str = None,
        max_size: int = None,
        max_files: int = None,
        enabled: bool = True
    ):
        self.log_dir = Path(log_dir) if log_dir else self.DEFAULT_LOG_DIR
        self.log_file = log_file or self.DEFAULT_LOG_FILE
        self.max_size = max_size or self.MAX_LOG_SIZE
        self.max_files = max_files or self.MAX_LOG_FILES
        self.enabled = enabled
        self.logger = logging.getLogger("audit")

        # Ensure log directory exists
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        """Full path to current log file."""
        return self.log_dir / self.log_file

    def _rotate_if_needed(self):
        """Rotate log file if it exceeds max size."""
        if not self.log_path.exists():
            return

        if self.log_path.stat().st_size < self.max_size:
            return

        # Rotate existing files
        for i in range(self.max_files - 1, 0, -1):
            old_path = self.log_dir / f"{self.log_file}.{i}"
            new_path = self.log_dir / f"{self.log_file}.{i + 1}"
            if old_path.exists():
                if i + 1 >= self.max_files:
                    old_path.unlink()  # Delete oldest
                else:
                    old_path.rename(new_path)

        # Rotate current file
        self.log_path.rename(self.log_dir / f"{self.log_file}.1")

    def log(self, entry: AuditEntry):
        """Write an audit entry to the log file."""
        if not self.enabled:
            return

        try:
            self._rotate_if_needed()

            # Use file locking for thread safety
            with open(self.log_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(entry.to_json() + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")

    def log_tool_invoked(
        self,
        session_id: str,
        tool_name: str,
        tool_args: dict,
        domain: str = None,
        operation: str = None,
        target: str = None
    ):
        """Log a tool invocation."""
        self.log(AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=AuditAction.TOOL_INVOKED.value,
            session_id=session_id,
            tool_name=tool_name,
            tool_args=tool_args,
            domain=domain,
            operation=operation,
            target=target
        ))

    def log_tool_completed(
        self,
        session_id: str,
        tool_name: str,
        tool_result: dict,
        duration_ms: int,
        success: bool
    ):
        """Log a tool completion."""
        self.log(AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=AuditAction.TOOL_COMPLETED.value,
            session_id=session_id,
            tool_name=tool_name,
            tool_result=self._sanitize_result(tool_result),
            duration_ms=duration_ms,
            success=success
        ))

    def log_tool_failed(
        self,
        session_id: str,
        tool_name: str,
        error: str,
        duration_ms: int = None
    ):
        """Log a tool failure."""
        self.log(AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=AuditAction.TOOL_FAILED.value,
            session_id=session_id,
            tool_name=tool_name,
            error=error,
            duration_ms=duration_ms,
            success=False
        ))

    def log_policy_check(
        self,
        session_id: str,
        domain: str,
        operation: str,
        target: str,
        level: str,
        source: str
    ):
        """Log a policy check."""
        self.log(AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=AuditAction.POLICY_CHECK.value,
            session_id=session_id,
            domain=domain,
            operation=operation,
            target=target,
            policy_level=level,
            policy_source=source
        ))

    def log_user_decision(
        self,
        session_id: str,
        approved: bool,
        domain: str,
        operation: str,
        target: str,
        response_time_ms: int = None
    ):
        """Log a user confirmation/denial decision."""
        action = AuditAction.USER_CONFIRMED if approved else AuditAction.USER_DENIED
        self.log(AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=action.value,
            session_id=session_id,
            domain=domain,
            operation=operation,
            target=target,
            duration_ms=response_time_ms,
            success=approved
        ))

    def log_event_triggered(
        self,
        trigger_id: str,
        event_type: str,
        event_source: str,
        metadata: dict = None
    ):
        """Log an event trigger."""
        self.log(AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=AuditAction.EVENT_TRIGGERED.value,
            trigger_id=trigger_id,
            event_type=event_type,
            event_source=event_source,
            metadata=metadata or {}
        ))

    def log_event_handled(
        self,
        trigger_id: str,
        event_type: str,
        success: bool,
        duration_ms: int = None,
        error: str = None
    ):
        """Log event handling completion."""
        self.log(AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=AuditAction.EVENT_HANDLED.value,
            trigger_id=trigger_id,
            event_type=event_type,
            success=success,
            duration_ms=duration_ms,
            error=error
        ))

    def log_profile_changed(
        self,
        session_id: str,
        old_profile: str,
        new_profile: str
    ):
        """Log a profile change."""
        self.log(AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=AuditAction.PROFILE_CHANGED.value,
            session_id=session_id,
            profile=new_profile,
            metadata={"old_profile": old_profile}
        ))

    def log_agent_lifecycle(self, action: AuditAction, metadata: dict = None):
        """Log agent start/stop events."""
        self.log(AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=action.value,
            metadata=metadata or {}
        ))

    def _sanitize_result(self, result: dict, max_length: int = 1000) -> dict:
        """Truncate large result values for logging."""
        if not result:
            return result

        sanitized = {}
        for key, value in result.items():
            if isinstance(value, str) and len(value) > max_length:
                sanitized[key] = value[:max_length] + f"... (truncated, {len(value)} chars)"
            elif isinstance(value, (list, dict)):
                str_value = json.dumps(value)
                if len(str_value) > max_length:
                    sanitized[key] = f"(truncated, {len(str_value)} chars)"
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        return sanitized

    def get_recent_entries(
        self,
        count: int = 50,
        action_filter: list[str] = None,
        session_id: str = None,
        since: datetime = None
    ) -> list[AuditEntry]:
        """
        Get recent audit entries.

        Args:
            count: Maximum number of entries to return
            action_filter: Only include these action types
            session_id: Only include entries for this session
            since: Only include entries after this time
        """
        if not self.log_path.exists():
            return []

        entries = []

        try:
            with open(self.log_path, "r") as f:
                # Read all lines (could be optimized for large files)
                lines = f.readlines()

            # Parse from newest to oldest
            for line in reversed(lines):
                if len(entries) >= count:
                    break

                try:
                    entry = AuditEntry.from_json(line.strip())

                    # Apply filters
                    if action_filter and entry.action not in action_filter:
                        continue
                    if session_id and entry.session_id != session_id:
                        continue
                    if since:
                        entry_time = datetime.fromisoformat(entry.timestamp)
                        if entry_time < since:
                            continue

                    entries.append(entry)

                except (json.JSONDecodeError, TypeError):
                    continue

        except Exception as e:
            self.logger.error(f"Failed to read audit log: {e}")

        return entries

    def get_summary(self, hours: int = 24) -> dict:
        """
        Get a summary of audit activity.

        Args:
            hours: Look back this many hours
        """
        since = datetime.now().replace(microsecond=0)
        since = since.replace(hour=since.hour - hours) if hours < 24 else None

        entries = self.get_recent_entries(count=10000, since=since)

        summary = {
            "period_hours": hours,
            "total_entries": len(entries),
            "actions": {},
            "tools_used": {},
            "policy_denials": 0,
            "user_denials": 0,
            "events_triggered": 0,
            "errors": 0
        }

        for entry in entries:
            # Count by action type
            summary["actions"][entry.action] = summary["actions"].get(entry.action, 0) + 1

            # Count tool usage
            if entry.tool_name:
                summary["tools_used"][entry.tool_name] = summary["tools_used"].get(entry.tool_name, 0) + 1

            # Count specific outcomes
            if entry.action == AuditAction.POLICY_DENIED.value:
                summary["policy_denials"] += 1
            if entry.action == AuditAction.USER_DENIED.value:
                summary["user_denials"] += 1
            if entry.action == AuditAction.EVENT_TRIGGERED.value:
                summary["events_triggered"] += 1
            if entry.error:
                summary["errors"] += 1

        return summary

    def clear(self):
        """Clear all audit logs (use with caution)."""
        if self.log_path.exists():
            self.log_path.unlink()

        # Also remove rotated files
        for i in range(1, self.max_files + 1):
            rotated = self.log_dir / f"{self.log_file}.{i}"
            if rotated.exists():
                rotated.unlink()


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def configure_audit_logger(
    log_dir: Path = None,
    enabled: bool = True,
    max_size: int = None,
    max_files: int = None
):
    """Configure the global audit logger."""
    global _audit_logger
    _audit_logger = AuditLogger(
        log_dir=log_dir,
        enabled=enabled,
        max_size=max_size,
        max_files=max_files
    )
    return _audit_logger
