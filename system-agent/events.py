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
    # File system events
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"

    # Scheduled events
    SCHEDULE = "schedule"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    LOW_DISK_SPACE = "low_disk_space"
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    PROCESS_STARTED = "process_started"
    PROCESS_STOPPED = "process_stopped"

    # Peripheral/hardware events
    USB_CONNECTED = "usb_connected"
    USB_DISCONNECTED = "usb_disconnected"
    NETWORK_CONNECTED = "network_connected"
    NETWORK_DISCONNECTED = "network_disconnected"
    NETWORK_CHANGED = "network_changed"
    BLUETOOTH_CONNECTED = "bluetooth_connected"
    BLUETOOTH_DISCONNECTED = "bluetooth_disconnected"
    POWER_AC_CONNECTED = "power_ac_connected"
    POWER_AC_DISCONNECTED = "power_ac_disconnected"
    POWER_LOW_BATTERY = "power_low_battery"
    DISPLAY_CONNECTED = "display_connected"
    DISPLAY_DISCONNECTED = "display_disconnected"
    AUDIO_DEVICE_CONNECTED = "audio_device_connected"
    AUDIO_DEVICE_DISCONNECTED = "audio_device_disconnected"

    # Custom events
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
            # File events
            EventType.FILE_CREATED: f"A new file was created: {self.source}\n{self._file_context()}",
            EventType.FILE_MODIFIED: f"A file was modified: {self.source}\n{self._file_context()}",
            EventType.FILE_DELETED: f"A file was deleted: {self.source}",
            EventType.FILE_MOVED: f"A file was moved from {self.data.get('src_path', '?')} to {self.source}",

            # Schedule events
            EventType.SCHEDULE: f"Scheduled task triggered: {self.data.get('task_name', self.source)}",

            # System events
            EventType.LOW_DISK_SPACE: f"Low disk space warning: {self.data.get('percent_free', '?')}% free on {self.source}",
            EventType.HIGH_CPU: f"High CPU usage detected: {self.data.get('percent', '?')}%",
            EventType.HIGH_MEMORY: f"High memory usage detected: {self.data.get('percent', '?')}%",

            # USB events
            EventType.USB_CONNECTED: f"USB device connected: {self._device_info()}",
            EventType.USB_DISCONNECTED: f"USB device disconnected: {self._device_info()}",

            # Network events
            EventType.NETWORK_CONNECTED: f"Network connected: {self.data.get('interface', self.source)} ({self.data.get('ssid', 'wired')})",
            EventType.NETWORK_DISCONNECTED: f"Network disconnected: {self.data.get('interface', self.source)}",
            EventType.NETWORK_CHANGED: f"Network changed: {self.data.get('interface', self.source)} - {self.data.get('reason', 'configuration changed')}",

            # Bluetooth events
            EventType.BLUETOOTH_CONNECTED: f"Bluetooth device connected: {self.data.get('device_name', self.source)}",
            EventType.BLUETOOTH_DISCONNECTED: f"Bluetooth device disconnected: {self.data.get('device_name', self.source)}",

            # Power events
            EventType.POWER_AC_CONNECTED: "Power adapter connected - now on AC power",
            EventType.POWER_AC_DISCONNECTED: "Power adapter disconnected - now on battery",
            EventType.POWER_LOW_BATTERY: f"Low battery warning: {self.data.get('percent', '?')}% remaining",

            # Display events
            EventType.DISPLAY_CONNECTED: f"Display connected: {self.data.get('display_name', self.source)}",
            EventType.DISPLAY_DISCONNECTED: f"Display disconnected: {self.data.get('display_name', self.source)}",

            # Audio events
            EventType.AUDIO_DEVICE_CONNECTED: f"Audio device connected: {self.data.get('device_name', self.source)}",
            EventType.AUDIO_DEVICE_DISCONNECTED: f"Audio device disconnected: {self.data.get('device_name', self.source)}",
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

    def _device_info(self) -> str:
        """Get formatted device information."""
        parts = []
        if self.data.get("vendor"):
            parts.append(self.data["vendor"])
        if self.data.get("product"):
            parts.append(self.data["product"])
        if self.data.get("device_type"):
            parts.append(f"({self.data['device_type']})")
        return " ".join(parts) if parts else self.source


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

    # Peripheral/device options
    device_types: list[str] = field(default_factory=list)  # ["usb_storage", "keyboard", etc.]
    interface_patterns: list[str] = field(default_factory=list)  # ["eth*", "wlan*", etc.]

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

    def matches_device(self, device_type: str, device_info: dict = None) -> bool:
        """Check if a device matches this trigger's device type filters."""
        # If no device_types filter, match all
        if not self.device_types:
            return True

        # Check if device_type is in the allowed list
        return device_type in self.device_types

    def matches_interface(self, interface: str) -> bool:
        """Check if a network interface matches this trigger's patterns."""
        # If no interface patterns, match all
        if not self.interface_patterns:
            return True

        for pattern in self.interface_patterns:
            if fnmatch.fnmatch(interface, pattern):
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

        # For cron, calculate next matching time
        return cls._next_cron_time(parsed, from_time)

    @classmethod
    def _next_cron_time(cls, parsed: dict, from_time: datetime) -> datetime:
        """Calculate next time matching cron expression."""
        # Start from the next minute
        next_time = from_time.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Try up to 366 days ahead (covers yearly schedules)
        for _ in range(366 * 24 * 60):
            if cls._matches_cron(parsed, next_time):
                return next_time
            next_time += timedelta(minutes=1)

        # Fallback if no match found (shouldn't happen with valid cron)
        return from_time + timedelta(hours=1)

    @classmethod
    def _matches_cron(cls, parsed: dict, dt: datetime) -> bool:
        """Check if datetime matches cron expression."""
        return (
            cls._matches_field(parsed["minute"], dt.minute, 0, 59) and
            cls._matches_field(parsed["hour"], dt.hour, 0, 23) and
            cls._matches_field(parsed["day"], dt.day, 1, 31) and
            cls._matches_field(parsed["month"], dt.month, 1, 12) and
            cls._matches_field(parsed["weekday"], dt.weekday(), 0, 6)  # Monday=0
        )

    @classmethod
    def _matches_field(cls, field: str, value: int, min_val: int, max_val: int) -> bool:
        """Check if a value matches a cron field expression."""
        # Wildcard matches everything
        if field == "*":
            return True

        # Handle */n (every n)
        if field.startswith("*/"):
            step = int(field[2:])
            return value % step == 0

        # Handle comma-separated values
        if "," in field:
            values = [int(v) for v in field.split(",")]
            return value in values

        # Handle ranges (e.g., 1-5)
        if "-" in field:
            start, end = field.split("-")
            return int(start) <= value <= int(end)

        # Simple numeric match
        try:
            return value == int(field)
        except ValueError:
            return False

    @classmethod
    def get_wait_seconds(cls, schedule: str, from_time: Optional[datetime] = None) -> float:
        """Calculate seconds until next scheduled run."""
        from_time = from_time or datetime.now()
        next_run = cls.get_next_run(schedule, from_time)
        delta = (next_run - from_time).total_seconds()
        return max(1, delta)  # At least 1 second


class PeripheralMonitor:
    """
    Monitors peripheral devices and hardware events.

    Uses /sys filesystem and udev for device monitoring on Linux.
    Falls back gracefully if pyudev is not available.
    """

    # USB device class codes
    USB_CLASSES = {
        "01": "audio",
        "02": "cdc_data",
        "03": "hid",  # Keyboards, mice
        "06": "image",  # Cameras, scanners
        "07": "printer",
        "08": "mass_storage",
        "09": "hub",
        "0a": "cdc_data",
        "0e": "video",
        "e0": "wireless",  # Bluetooth adapters
        "ef": "misc",
        "ff": "vendor_specific",
    }

    def __init__(self, event_queue: asyncio.Queue, triggers: dict):
        self.event_queue = event_queue
        self.triggers = triggers
        self.logger = logging.getLogger("peripheral-monitor")
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._udev_observer = None

        # Track current state for change detection
        self._known_usb_devices: set = set()
        self._known_network_interfaces: dict = {}
        self._last_power_state: Optional[bool] = None  # True = on AC

        # Try to import pyudev
        try:
            import pyudev
            self._pyudev = pyudev
            self._udev_available = True
        except ImportError:
            self._pyudev = None
            self._udev_available = False
            self.logger.info("pyudev not available, using polling for USB events")

    def _get_relevant_triggers(self, event_type: EventType) -> list:
        """Get triggers that care about this event type."""
        return [
            t for t in self.triggers.values()
            if t.enabled and event_type in t.event_types
        ]

    async def start(self):
        """Start peripheral monitoring."""
        if self.running:
            return

        self.running = True

        # Initialize known state
        self._known_usb_devices = self._get_current_usb_devices()
        self._known_network_interfaces = self._get_network_state()
        self._last_power_state = self._get_power_state()

        # Start the monitor task
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Peripheral monitor started")

    async def stop(self):
        """Stop peripheral monitoring."""
        self.running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        self.logger.info("Peripheral monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop - polls for changes."""
        while self.running:
            try:
                await self._check_usb_changes()
                await self._check_network_changes()
                await self._check_power_changes()

                # Poll every 2 seconds
                await asyncio.sleep(2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Peripheral monitor error: {e}")
                await asyncio.sleep(5)

    def _get_current_usb_devices(self) -> set:
        """Get set of currently connected USB devices."""
        devices = set()
        usb_path = Path("/sys/bus/usb/devices")

        if not usb_path.exists():
            return devices

        for device_dir in usb_path.iterdir():
            if not device_dir.is_dir():
                continue

            # Skip hub interfaces (contain ":")
            if ":" in device_dir.name:
                continue

            try:
                vendor_file = device_dir / "idVendor"
                product_file = device_dir / "idProduct"

                if vendor_file.exists() and product_file.exists():
                    vendor = vendor_file.read_text().strip()
                    product = product_file.read_text().strip()
                    device_id = f"{vendor}:{product}"
                    devices.add(device_id)
            except (OSError, IOError):
                continue

        return devices

    def _get_usb_device_info(self, device_id: str) -> dict:
        """Get detailed info about a USB device."""
        usb_path = Path("/sys/bus/usb/devices")
        vendor_id, product_id = device_id.split(":")

        for device_dir in usb_path.iterdir():
            if ":" in device_dir.name or not device_dir.is_dir():
                continue

            try:
                vendor_file = device_dir / "idVendor"
                product_file = device_dir / "idProduct"

                if not (vendor_file.exists() and product_file.exists()):
                    continue

                if vendor_file.read_text().strip() != vendor_id:
                    continue
                if product_file.read_text().strip() != product_id:
                    continue

                # Found the device, get more info
                info = {
                    "vendor_id": vendor_id,
                    "product_id": product_id,
                }

                # Get manufacturer name
                mfr_file = device_dir / "manufacturer"
                if mfr_file.exists():
                    info["vendor"] = mfr_file.read_text().strip()

                # Get product name
                prod_name_file = device_dir / "product"
                if prod_name_file.exists():
                    info["product"] = prod_name_file.read_text().strip()

                # Get device class
                class_file = device_dir / "bDeviceClass"
                if class_file.exists():
                    class_code = class_file.read_text().strip().lower()
                    info["device_type"] = self.USB_CLASSES.get(class_code, "unknown")

                return info

            except (OSError, IOError):
                continue

        return {"vendor_id": vendor_id, "product_id": product_id}

    async def _check_usb_changes(self):
        """Check for USB device changes."""
        current_devices = self._get_current_usb_devices()

        # Check for new devices
        new_devices = current_devices - self._known_usb_devices
        for device_id in new_devices:
            triggers = self._get_relevant_triggers(EventType.USB_CONNECTED)
            for trigger in triggers:
                if trigger.can_trigger():
                    device_info = self._get_usb_device_info(device_id)

                    if trigger.matches_device(device_info.get("device_type", "unknown"), device_info):
                        trigger.mark_triggered()
                        event = Event(
                            event_type=EventType.USB_CONNECTED,
                            source=device_id,
                            data={**device_info, "prompt": trigger.prompt},
                            trigger_id=trigger.id
                        )
                        await self.event_queue.put(event)
                        self.logger.info(f"USB connected: {device_info.get('product', device_id)}")

        # Check for removed devices
        removed_devices = self._known_usb_devices - current_devices
        for device_id in removed_devices:
            triggers = self._get_relevant_triggers(EventType.USB_DISCONNECTED)
            for trigger in triggers:
                if trigger.can_trigger():
                    trigger.mark_triggered()
                    event = Event(
                        event_type=EventType.USB_DISCONNECTED,
                        source=device_id,
                        data={"prompt": trigger.prompt},
                        trigger_id=trigger.id
                    )
                    await self.event_queue.put(event)
                    self.logger.info(f"USB disconnected: {device_id}")

        self._known_usb_devices = current_devices

    def _get_network_state(self) -> dict:
        """Get current network interface states."""
        state = {}
        net_path = Path("/sys/class/net")

        if not net_path.exists():
            return state

        for iface_dir in net_path.iterdir():
            iface_name = iface_dir.name
            if iface_name == "lo":  # Skip loopback
                continue

            try:
                # Check operstate (up/down)
                operstate_file = iface_dir / "operstate"
                if operstate_file.exists():
                    operstate = operstate_file.read_text().strip()
                    state[iface_name] = {
                        "state": operstate,
                        "is_up": operstate in ("up", "unknown"),
                    }

                    # Try to get carrier (actual link state)
                    carrier_file = iface_dir / "carrier"
                    if carrier_file.exists():
                        try:
                            carrier = int(carrier_file.read_text().strip())
                            state[iface_name]["has_carrier"] = carrier == 1
                        except (ValueError, IOError):
                            pass

            except (OSError, IOError):
                continue

        return state

    async def _check_network_changes(self):
        """Check for network interface changes."""
        current_state = self._get_network_state()

        for iface, info in current_state.items():
            old_info = self._known_network_interfaces.get(iface, {})

            # Check for new connection
            if info.get("is_up") and not old_info.get("is_up"):
                triggers = self._get_relevant_triggers(EventType.NETWORK_CONNECTED)
                for trigger in triggers:
                    if trigger.can_trigger() and trigger.matches_interface(iface):
                        trigger.mark_triggered()
                        event = Event(
                            event_type=EventType.NETWORK_CONNECTED,
                            source=iface,
                            data={"interface": iface, "prompt": trigger.prompt},
                            trigger_id=trigger.id
                        )
                        await self.event_queue.put(event)
                        self.logger.info(f"Network connected: {iface}")

            # Check for disconnection
            elif not info.get("is_up") and old_info.get("is_up"):
                triggers = self._get_relevant_triggers(EventType.NETWORK_DISCONNECTED)
                for trigger in triggers:
                    if trigger.can_trigger() and trigger.matches_interface(iface):
                        trigger.mark_triggered()
                        event = Event(
                            event_type=EventType.NETWORK_DISCONNECTED,
                            source=iface,
                            data={"interface": iface, "prompt": trigger.prompt},
                            trigger_id=trigger.id
                        )
                        await self.event_queue.put(event)
                        self.logger.info(f"Network disconnected: {iface}")

        # Check for interfaces that disappeared
        for iface in set(self._known_network_interfaces.keys()) - set(current_state.keys()):
            triggers = self._get_relevant_triggers(EventType.NETWORK_DISCONNECTED)
            for trigger in triggers:
                if trigger.can_trigger() and trigger.matches_interface(iface):
                    trigger.mark_triggered()
                    event = Event(
                        event_type=EventType.NETWORK_DISCONNECTED,
                        source=iface,
                        data={"interface": iface, "reason": "interface_removed", "prompt": trigger.prompt},
                        trigger_id=trigger.id
                    )
                    await self.event_queue.put(event)

        self._known_network_interfaces = current_state

    def _get_power_state(self) -> Optional[bool]:
        """Get current power state (True = AC, False = battery, None = unknown)."""
        # Check common power supply paths
        power_paths = [
            Path("/sys/class/power_supply/AC/online"),
            Path("/sys/class/power_supply/AC0/online"),
            Path("/sys/class/power_supply/ACAD/online"),
        ]

        for path in power_paths:
            if path.exists():
                try:
                    return int(path.read_text().strip()) == 1
                except (ValueError, IOError):
                    continue

        return None

    def _get_battery_level(self) -> Optional[int]:
        """Get battery percentage if available."""
        battery_paths = [
            Path("/sys/class/power_supply/BAT0/capacity"),
            Path("/sys/class/power_supply/BAT1/capacity"),
        ]

        for path in battery_paths:
            if path.exists():
                try:
                    return int(path.read_text().strip())
                except (ValueError, IOError):
                    continue

        return None

    async def _check_power_changes(self):
        """Check for power state changes."""
        current_state = self._get_power_state()

        if current_state is None or self._last_power_state is None:
            self._last_power_state = current_state
            return

        if current_state != self._last_power_state:
            if current_state:  # Now on AC
                event_type = EventType.POWER_AC_CONNECTED
            else:  # Now on battery
                event_type = EventType.POWER_AC_DISCONNECTED

            triggers = self._get_relevant_triggers(event_type)
            for trigger in triggers:
                if trigger.can_trigger():
                    trigger.mark_triggered()
                    event = Event(
                        event_type=event_type,
                        source="power",
                        data={"on_ac": current_state, "prompt": trigger.prompt},
                        trigger_id=trigger.id
                    )
                    await self.event_queue.put(event)
                    self.logger.info(f"Power state changed: {'AC' if current_state else 'Battery'}")

        # Also check for low battery
        if not current_state:  # On battery
            battery_level = self._get_battery_level()
            if battery_level is not None:
                triggers = self._get_relevant_triggers(EventType.POWER_LOW_BATTERY)
                for trigger in triggers:
                    threshold = trigger.threshold or 20.0
                    if battery_level <= threshold and trigger.can_trigger():
                        trigger.mark_triggered()
                        event = Event(
                            event_type=EventType.POWER_LOW_BATTERY,
                            source="battery",
                            data={"percent": battery_level, "prompt": trigger.prompt},
                            trigger_id=trigger.id
                        )
                        await self.event_queue.put(event)
                        self.logger.info(f"Low battery: {battery_level}%")

        self._last_power_state = current_state


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
        self._peripheral_monitor: Optional[PeripheralMonitor] = None

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
                device_types=tconfig.get("device_types", []),
                interface_patterns=tconfig.get("interface_patterns", []),
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

        # Start peripheral monitor
        await self._start_peripheral_monitor()

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

        # Stop peripheral monitor
        if self._peripheral_monitor:
            await self._peripheral_monitor.stop()
            self._peripheral_monitor = None

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
                # Calculate wait time until next scheduled run
                wait_seconds = ScheduleParser.get_wait_seconds(trigger.schedule)
                self.logger.debug(f"Trigger {trigger.id}: waiting {wait_seconds:.0f}s until next run")

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

    async def _start_peripheral_monitor(self):
        """Start peripheral device monitoring."""
        peripheral_events = {
            EventType.USB_CONNECTED, EventType.USB_DISCONNECTED,
            EventType.NETWORK_CONNECTED, EventType.NETWORK_DISCONNECTED, EventType.NETWORK_CHANGED,
            EventType.BLUETOOTH_CONNECTED, EventType.BLUETOOTH_DISCONNECTED,
            EventType.POWER_AC_CONNECTED, EventType.POWER_AC_DISCONNECTED, EventType.POWER_LOW_BATTERY,
            EventType.DISPLAY_CONNECTED, EventType.DISPLAY_DISCONNECTED,
            EventType.AUDIO_DEVICE_CONNECTED, EventType.AUDIO_DEVICE_DISCONNECTED,
        }

        # Check if any triggers need peripheral monitoring
        needs_monitor = any(
            peripheral_events.intersection(t.event_types)
            for t in self.triggers.values()
            if t.enabled
        )

        if needs_monitor:
            self._peripheral_monitor = PeripheralMonitor(self.event_queue, self.triggers)
            await self._peripheral_monitor.start()
            self.logger.info("Started peripheral monitor")

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
                    "device_types": t.device_types if t.device_types else None,
                    "interface_patterns": t.interface_patterns if t.interface_patterns else None,
                }
                for tid, t in self.triggers.items()
            },
            "file_watchers": len(self.file_observers),
            "schedule_tasks": len(self._schedule_tasks),
            "peripheral_monitor": self._peripheral_monitor is not None and self._peripheral_monitor.running,
            "queue_size": self.event_queue.qsize(),
        }
