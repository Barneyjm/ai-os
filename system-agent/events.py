"""
Event System for AI-OS

Provides proactive agency through file watching, scheduled tasks,
and system event hooks that trigger the AI agent automatically.
"""

import asyncio
import logging
import os
import re
import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
import json

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object
    FileSystemEvent = None


class EventType(Enum):
    """Types of events that can trigger the agent."""
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    SCHEDULE = "schedule"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    LOW_DISK_SPACE = "low_disk_space"
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    PROCESS_STARTED = "process_started"
    PROCESS_STOPPED = "process_stopped"
    CUSTOM = "custom"


@dataclass
class Event:
    """Represents an event that occurred."""
    event_type: EventType
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict = field(default_factory=dict)
    trigger_id: Optional[str] = None

    def to_prompt(self) -> str:
        """Convert event to a prompt for the agent."""
        prompts = {
            EventType.FILE_CREATED: f"A new file was created: {self.source}\n{self._file_context()}",
            EventType.FILE_MODIFIED: f"A file was modified: {self.source}\n{self._file_context()}",
            EventType.FILE_DELETED: f"A file was deleted: {self.source}",
            EventType.FILE_MOVED: f"A file was moved from {self.data.get('src_path', '?')} to {self.source}",
            EventType.SCHEDULE: f"Scheduled task triggered: {self.data.get('task_name', self.source)}",
            EventType.LOW_DISK_SPACE: f"Low disk space warning: {self.data.get('percent_free', '?')}% free on {self.source}",
            EventType.HIGH_CPU: f"High CPU usage detected: {self.data.get('percent', '?')}%",
            EventType.HIGH_MEMORY: f"High memory usage detected: {self.data.get('percent', '?')}%",
        }
        base = prompts.get(self.event_type, f"Event occurred: {self.event_type.value} on {self.source}")

        if self.data.get("prompt"):
            base += f"\n\nInstructions: {self.data['prompt']}"

        return base

    def _file_context(self) -> str:
        """Get context about a file event."""
        parts = []
        if self.data.get("is_directory"):
            parts.append("(directory)")
        if self.data.get("size"):
            parts.append(f"Size: {self.data['size']} bytes")
        return " ".join(parts)


@dataclass
class EventTrigger:
    """Defines when and how to trigger on events."""
    id: str
    event_types: list[EventType]
    enabled: bool = True

    # File watching options
    watch_path: Optional[str] = None
    file_patterns: list[str] = field(default_factory=lambda: ["*"])
    recursive: bool = True
    ignore_patterns: list[str] = field(default_factory=lambda: [".*", "__pycache__", "*.pyc", ".git"])

    # Schedule options (cron-like)
    schedule: Optional[str] = None  # "*/5 * * * *" or "@hourly", "@daily", etc.

    # System monitoring options
    threshold: Optional[float] = None  # For CPU/memory/disk thresholds

    # Action configuration
    prompt: Optional[str] = None  # Custom prompt to send to agent
    auto_approve: bool = False  # Skip policy check for this trigger
    cooldown_seconds: int = 60  # Minimum time between triggers

    # Runtime state
    last_triggered: Optional[datetime] = field(default=None, repr=False)

    def matches_file(self, path: str) -> bool:
        """Check if a file path matches this trigger's patterns."""
        filename = os.path.basename(path)

        # Check ignore patterns first
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(path, f"*/{pattern}/*"):
                return False

        # Check include patterns
        for pattern in self.file_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def can_trigger(self) -> bool:
        """Check if trigger is ready (respects cooldown)."""
        if not self.enabled:
            return False
        if self.last_triggered is None:
            return True
        elapsed = (datetime.now() - self.last_triggered).total_seconds()
        return elapsed >= self.cooldown_seconds

    def mark_triggered(self):
        """Mark this trigger as having fired."""
        self.last_triggered = datetime.now()


class EventHandler(ABC):
    """Base class for event handlers."""

    @abstractmethod
    async def handle(self, event: Event) -> dict:
        """Handle an event. Returns result dict."""
        pass


class AgentEventHandler(EventHandler):
    """Sends events to the SystemAgent for processing."""

    def __init__(self, agent_callback: Callable):
        self.agent_callback = agent_callback
        self.logger = logging.getLogger("event-handler")

    async def handle(self, event: Event) -> dict:
        """Send event to agent as a message."""
        prompt = event.to_prompt()
        self.logger.info(f"Handling event {event.event_type.value}: {event.source}")

        try:
            result = await self.agent_callback(prompt, event)
            return {"success": True, "result": result}
        except Exception as e:
            self.logger.error(f"Error handling event: {e}")
            return {"success": False, "error": str(e)}


class FileWatcherHandler(FileSystemEventHandler):
    """Watchdog handler that creates Events from filesystem changes."""

    def __init__(self, trigger: EventTrigger, event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.trigger = trigger
        self.event_queue = event_queue
        self.loop = loop
        self.logger = logging.getLogger("file-watcher")

    def _create_event(self, event_type: EventType, src_path: str, **extra_data) -> Optional[Event]:
        """Create an Event if the path matches the trigger."""
        if not self.trigger.matches_file(src_path):
            return None
        if not self.trigger.can_trigger():
            self.logger.debug(f"Trigger {self.trigger.id} in cooldown, skipping")
            return None

        self.trigger.mark_triggered()

        data = {
            "is_directory": extra_data.get("is_directory", False),
            "prompt": self.trigger.prompt,
            **extra_data
        }

        return Event(
            event_type=event_type,
            source=src_path,
            data=data,
            trigger_id=self.trigger.id
        )

    def _queue_event(self, event: Optional[Event]):
        """Thread-safe way to add event to async queue."""
        if event:
            self.loop.call_soon_threadsafe(
                lambda: self.event_queue.put_nowait(event)
            )

    def on_created(self, event: FileSystemEvent):
        if EventType.FILE_CREATED in self.trigger.event_types:
            e = self._create_event(
                EventType.FILE_CREATED,
                event.src_path,
                is_directory=event.is_directory
            )
            self._queue_event(e)

    def on_modified(self, event: FileSystemEvent):
        if EventType.FILE_MODIFIED in self.trigger.event_types:
            e = self._create_event(
                EventType.FILE_MODIFIED,
                event.src_path,
                is_directory=event.is_directory
            )
            self._queue_event(e)

    def on_deleted(self, event: FileSystemEvent):
        if EventType.FILE_DELETED in self.trigger.event_types:
            e = self._create_event(
                EventType.FILE_DELETED,
                event.src_path,
                is_directory=event.is_directory
            )
            self._queue_event(e)

    def on_moved(self, event: FileSystemEvent):
        if EventType.FILE_MOVED in self.trigger.event_types:
            e = self._create_event(
                EventType.FILE_MOVED,
                event.dest_path,
                is_directory=event.is_directory,
                src_path=event.src_path
            )
            self._queue_event(e)


class ScheduleParser:
    """Simple schedule parser supporting cron-like syntax and shortcuts."""

    SHORTCUTS = {
        "@yearly": "0 0 1 1 *",
        "@monthly": "0 0 1 * *",
        "@weekly": "0 0 * * 0",
        "@daily": "0 0 * * *",
        "@hourly": "0 * * * *",
        "@minutely": "* * * * *",
    }

    @classmethod
    def parse(cls, schedule: str) -> dict:
        """Parse a schedule string into components."""
        schedule = schedule.strip()

        # Handle shortcuts
        if schedule.startswith("@"):
            if schedule in cls.SHORTCUTS:
                schedule = cls.SHORTCUTS[schedule]
            elif schedule.startswith("@every_"):
                # @every_5m, @every_1h, @every_30s
                match = re.match(r"@every_(\d+)([smhd])", schedule)
                if match:
                    amount, unit = int(match.group(1)), match.group(2)
                    return {"type": "interval", "amount": amount, "unit": unit}
            else:
                raise ValueError(f"Unknown schedule shortcut: {schedule}")

        # Parse cron expression (minute hour day month weekday)
        parts = schedule.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {schedule}")

        return {
            "type": "cron",
            "minute": parts[0],
            "hour": parts[1],
            "day": parts[2],
            "month": parts[3],
            "weekday": parts[4]
        }

    @classmethod
    def get_next_run(cls, schedule: str, from_time: Optional[datetime] = None) -> datetime:
        """Calculate the next run time for a schedule."""
        from_time = from_time or datetime.now()
        parsed = cls.parse(schedule)

        if parsed["type"] == "interval":
            units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
            seconds = parsed["amount"] * units[parsed["unit"]]
            return from_time + timedelta(seconds=seconds)

        # For cron, do simple next-minute calculation for now
        # A full cron implementation would be more complex
        return from_time + timedelta(minutes=1)


class EventManager:
    """Manages event triggers and dispatches events to handlers."""

    def __init__(self):
        self.triggers: dict[str, EventTrigger] = {}
        self.handlers: list[EventHandler] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.file_observers: list = []
        self.running = False
        self.logger = logging.getLogger("event-manager")
        self._schedule_tasks: dict[str, asyncio.Task] = {}
        self._system_monitor_task: Optional[asyncio.Task] = None

    def register_trigger(self, trigger: EventTrigger):
        """Register a new event trigger."""
        self.triggers[trigger.id] = trigger
        self.logger.info(f"Registered trigger: {trigger.id}")

    def unregister_trigger(self, trigger_id: str):
        """Remove a trigger."""
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            self.logger.info(f"Unregistered trigger: {trigger_id}")

    def add_handler(self, handler: EventHandler):
        """Add an event handler."""
        self.handlers.append(handler)

    def load_triggers_from_config(self, config: dict):
        """Load triggers from configuration dict."""
        events_config = config.get("events", {})
        triggers_config = events_config.get("triggers", [])

        for tconfig in triggers_config:
            event_types = [EventType(et) for et in tconfig.get("event_types", [])]
            trigger = EventTrigger(
                id=tconfig["id"],
                event_types=event_types,
                enabled=tconfig.get("enabled", True),
                watch_path=tconfig.get("watch_path"),
                file_patterns=tconfig.get("file_patterns", ["*"]),
                recursive=tconfig.get("recursive", True),
                ignore_patterns=tconfig.get("ignore_patterns", [".*", "__pycache__", "*.pyc", ".git"]),
                schedule=tconfig.get("schedule"),
                threshold=tconfig.get("threshold"),
                prompt=tconfig.get("prompt"),
                auto_approve=tconfig.get("auto_approve", False),
                cooldown_seconds=tconfig.get("cooldown_seconds", 60),
            )
            self.register_trigger(trigger)

    async def start(self):
        """Start the event manager."""
        if self.running:
            return

        self.running = True
        self.logger.info("Starting event manager")

        # Start file watchers
        await self._start_file_watchers()

        # Start schedule tasks
        await self._start_schedule_tasks()

        # Start system monitor
        await self._start_system_monitor()

        # Start event processing loop
        asyncio.create_task(self._process_events())

    async def stop(self):
        """Stop the event manager."""
        self.running = False
        self.logger.info("Stopping event manager")

        # Stop file watchers
        for observer in self.file_observers:
            observer.stop()
            observer.join(timeout=2)
        self.file_observers.clear()

        # Cancel schedule tasks
        for task in self._schedule_tasks.values():
            task.cancel()
        self._schedule_tasks.clear()

        # Cancel system monitor
        if self._system_monitor_task:
            self._system_monitor_task.cancel()
            self._system_monitor_task = None

    async def _start_file_watchers(self):
        """Start file system watchers for triggers that need them."""
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("watchdog not installed, file watching disabled")
            return

        loop = asyncio.get_event_loop()

        for trigger in self.triggers.values():
            if trigger.watch_path and trigger.enabled:
                file_events = {EventType.FILE_CREATED, EventType.FILE_MODIFIED,
                              EventType.FILE_DELETED, EventType.FILE_MOVED}
                if not file_events.intersection(trigger.event_types):
                    continue

                path = os.path.expanduser(trigger.watch_path)
                if not os.path.exists(path):
                    self.logger.warning(f"Watch path does not exist: {path}")
                    continue

                handler = FileWatcherHandler(trigger, self.event_queue, loop)
                observer = Observer()
                observer.schedule(handler, path, recursive=trigger.recursive)
                observer.start()
                self.file_observers.append(observer)
                self.logger.info(f"Started file watcher for {path}")

    async def _start_schedule_tasks(self):
        """Start scheduled trigger tasks."""
        for trigger in self.triggers.values():
            if trigger.schedule and trigger.enabled:
                task = asyncio.create_task(self._run_schedule(trigger))
                self._schedule_tasks[trigger.id] = task
                self.logger.info(f"Started schedule for trigger {trigger.id}: {trigger.schedule}")

    async def _run_schedule(self, trigger: EventTrigger):
        """Run a scheduled trigger."""
        while self.running:
            try:
                # Calculate next run time
                parsed = ScheduleParser.parse(trigger.schedule)

                if parsed["type"] == "interval":
                    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
                    wait_seconds = parsed["amount"] * units[parsed["unit"]]
                else:
                    # Simple minute-based for cron
                    wait_seconds = 60

                await asyncio.sleep(wait_seconds)

                if trigger.can_trigger():
                    trigger.mark_triggered()
                    event = Event(
                        event_type=EventType.SCHEDULE,
                        source=trigger.id,
                        data={
                            "task_name": trigger.id,
                            "prompt": trigger.prompt,
                            "schedule": trigger.schedule
                        },
                        trigger_id=trigger.id
                    )
                    await self.event_queue.put(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Schedule error for {trigger.id}: {e}")
                await asyncio.sleep(60)

    async def _start_system_monitor(self):
        """Start system resource monitoring."""
        system_events = {EventType.LOW_DISK_SPACE, EventType.HIGH_CPU, EventType.HIGH_MEMORY}

        # Check if any triggers need system monitoring
        needs_monitor = any(
            system_events.intersection(t.event_types)
            for t in self.triggers.values()
            if t.enabled
        )

        if needs_monitor:
            self._system_monitor_task = asyncio.create_task(self._monitor_system())
            self.logger.info("Started system monitor")

    async def _monitor_system(self):
        """Monitor system resources and trigger events."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for trigger in self.triggers.values():
                    if not trigger.enabled or not trigger.can_trigger():
                        continue

                    threshold = trigger.threshold or 90.0

                    if EventType.LOW_DISK_SPACE in trigger.event_types:
                        await self._check_disk_space(trigger, threshold)

                    if EventType.HIGH_CPU in trigger.event_types:
                        await self._check_cpu(trigger, threshold)

                    if EventType.HIGH_MEMORY in trigger.event_types:
                        await self._check_memory(trigger, threshold)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System monitor error: {e}")

    async def _check_disk_space(self, trigger: EventTrigger, threshold: float):
        """Check disk space and trigger if low."""
        try:
            path = trigger.watch_path or "/"
            stat = os.statvfs(path)
            percent_free = (stat.f_bavail / stat.f_blocks) * 100

            if percent_free < (100 - threshold):
                trigger.mark_triggered()
                event = Event(
                    event_type=EventType.LOW_DISK_SPACE,
                    source=path,
                    data={"percent_free": round(percent_free, 1), "prompt": trigger.prompt},
                    trigger_id=trigger.id
                )
                await self.event_queue.put(event)
        except Exception as e:
            self.logger.debug(f"Disk check error: {e}")

    async def _check_cpu(self, trigger: EventTrigger, threshold: float):
        """Check CPU usage."""
        try:
            with open("/proc/loadavg", "r") as f:
                load = float(f.read().split()[0])
                cpu_count = os.cpu_count() or 1
                percent = (load / cpu_count) * 100

                if percent > threshold:
                    trigger.mark_triggered()
                    event = Event(
                        event_type=EventType.HIGH_CPU,
                        source="system",
                        data={"percent": round(percent, 1), "load": load, "prompt": trigger.prompt},
                        trigger_id=trigger.id
                    )
                    await self.event_queue.put(event)
        except Exception as e:
            self.logger.debug(f"CPU check error: {e}")

    async def _check_memory(self, trigger: EventTrigger, threshold: float):
        """Check memory usage."""
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(":")] = int(parts[1])

                total = meminfo.get("MemTotal", 1)
                available = meminfo.get("MemAvailable", total)
                percent_used = ((total - available) / total) * 100

                if percent_used > threshold:
                    trigger.mark_triggered()
                    event = Event(
                        event_type=EventType.HIGH_MEMORY,
                        source="system",
                        data={"percent": round(percent_used, 1), "prompt": trigger.prompt},
                        trigger_id=trigger.id
                    )
                    await self.event_queue.put(event)
        except Exception as e:
            self.logger.debug(f"Memory check error: {e}")

    async def _process_events(self):
        """Process events from the queue."""
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                self.logger.info(f"Processing event: {event.event_type.value} from {event.source}")

                for handler in self.handlers:
                    try:
                        await handler.handle(event)
                    except Exception as e:
                        self.logger.error(f"Handler error: {e}")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")

    async def emit(self, event: Event):
        """Manually emit an event."""
        await self.event_queue.put(event)

    def get_status(self) -> dict:
        """Get current status of the event manager."""
        return {
            "running": self.running,
            "triggers": {
                tid: {
                    "enabled": t.enabled,
                    "event_types": [et.value for et in t.event_types],
                    "last_triggered": t.last_triggered.isoformat() if t.last_triggered else None,
                    "watch_path": t.watch_path,
                    "schedule": t.schedule,
                }
                for tid, t in self.triggers.items()
            },
            "file_watchers": len(self.file_observers),
            "schedule_tasks": len(self._schedule_tasks),
            "queue_size": self.event_queue.qsize(),
        }
