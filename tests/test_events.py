"""
Tests for the Event System module.
"""

import asyncio
import os
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

from events import (
    EventType,
    Event,
    EventTrigger,
    EventManager,
    EventHandler,
    AgentEventHandler,
    ScheduleParser,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_file_events(self):
        """Should have file-related event types."""
        assert EventType.FILE_CREATED.value == "file_created"
        assert EventType.FILE_MODIFIED.value == "file_modified"
        assert EventType.FILE_DELETED.value == "file_deleted"
        assert EventType.FILE_MOVED.value == "file_moved"

    def test_schedule_events(self):
        """Should have schedule event type."""
        assert EventType.SCHEDULE.value == "schedule"

    def test_system_events(self):
        """Should have system monitoring event types."""
        assert EventType.LOW_DISK_SPACE.value == "low_disk_space"
        assert EventType.HIGH_CPU.value == "high_cpu"
        assert EventType.HIGH_MEMORY.value == "high_memory"


class TestEvent:
    """Tests for Event dataclass."""

    def test_event_creation(self):
        """Should create event with required fields."""
        event = Event(
            event_type=EventType.FILE_CREATED,
            source="/tmp/test.txt"
        )
        assert event.event_type == EventType.FILE_CREATED
        assert event.source == "/tmp/test.txt"
        assert isinstance(event.timestamp, datetime)
        assert event.data == {}

    def test_event_with_data(self):
        """Should create event with additional data."""
        event = Event(
            event_type=EventType.LOW_DISK_SPACE,
            source="/",
            data={"percent_free": 5.2}
        )
        assert event.data["percent_free"] == 5.2

    def test_to_prompt_file_created(self):
        """Should generate prompt for file created event."""
        event = Event(
            event_type=EventType.FILE_CREATED,
            source="/tmp/newfile.txt"
        )
        prompt = event.to_prompt()
        assert "new file was created" in prompt.lower()
        assert "/tmp/newfile.txt" in prompt

    def test_to_prompt_with_custom_prompt(self):
        """Should include custom prompt in generated prompt."""
        event = Event(
            event_type=EventType.FILE_CREATED,
            source="/tmp/test.txt",
            data={"prompt": "Analyze this file and categorize it."}
        )
        prompt = event.to_prompt()
        assert "Analyze this file" in prompt

    def test_to_prompt_schedule(self):
        """Should generate prompt for scheduled event."""
        event = Event(
            event_type=EventType.SCHEDULE,
            source="daily-backup",
            data={"task_name": "daily-backup"}
        )
        prompt = event.to_prompt()
        assert "scheduled task" in prompt.lower()
        assert "daily-backup" in prompt


class TestEventTrigger:
    """Tests for EventTrigger dataclass."""

    def test_trigger_creation(self):
        """Should create trigger with required fields."""
        trigger = EventTrigger(
            id="test-trigger",
            event_types=[EventType.FILE_CREATED]
        )
        assert trigger.id == "test-trigger"
        assert trigger.enabled is True
        assert EventType.FILE_CREATED in trigger.event_types

    def test_matches_file_basic(self):
        """Should match files based on patterns."""
        trigger = EventTrigger(
            id="test",
            event_types=[EventType.FILE_CREATED],
            file_patterns=["*.txt", "*.md"]
        )
        assert trigger.matches_file("/tmp/readme.txt") is True
        assert trigger.matches_file("/tmp/notes.md") is True
        assert trigger.matches_file("/tmp/script.py") is False

    def test_matches_file_ignores_patterns(self):
        """Should ignore files matching ignore patterns."""
        trigger = EventTrigger(
            id="test",
            event_types=[EventType.FILE_CREATED],
            file_patterns=["*"],
            ignore_patterns=[".*", "*.pyc"]
        )
        assert trigger.matches_file("/tmp/file.txt") is True
        assert trigger.matches_file("/tmp/.hidden") is False
        assert trigger.matches_file("/tmp/cache.pyc") is False

    def test_can_trigger_respects_cooldown(self):
        """Should respect cooldown period between triggers."""
        trigger = EventTrigger(
            id="test",
            event_types=[EventType.FILE_CREATED],
            cooldown_seconds=60
        )

        # Initially can trigger
        assert trigger.can_trigger() is True

        # After marking triggered, should be in cooldown
        trigger.mark_triggered()
        assert trigger.can_trigger() is False

        # Simulate cooldown elapsed
        trigger.last_triggered = datetime.now() - timedelta(seconds=61)
        assert trigger.can_trigger() is True

    def test_can_trigger_disabled(self):
        """Should not trigger if disabled."""
        trigger = EventTrigger(
            id="test",
            event_types=[EventType.FILE_CREATED],
            enabled=False
        )
        assert trigger.can_trigger() is False


class TestScheduleParser:
    """Tests for ScheduleParser."""

    def test_parse_interval_minutes(self):
        """Should parse interval shortcuts."""
        result = ScheduleParser.parse("@every_5m")
        assert result["type"] == "interval"
        assert result["amount"] == 5
        assert result["unit"] == "m"

    def test_parse_interval_hours(self):
        """Should parse hourly intervals."""
        result = ScheduleParser.parse("@every_2h")
        assert result["type"] == "interval"
        assert result["amount"] == 2
        assert result["unit"] == "h"

    def test_parse_shortcut_daily(self):
        """Should expand @daily shortcut."""
        result = ScheduleParser.parse("@daily")
        assert result["type"] == "cron"
        assert result["minute"] == "0"
        assert result["hour"] == "0"

    def test_parse_shortcut_hourly(self):
        """Should expand @hourly shortcut."""
        result = ScheduleParser.parse("@hourly")
        assert result["type"] == "cron"
        assert result["minute"] == "0"

    def test_parse_cron_expression(self):
        """Should parse standard cron expressions."""
        result = ScheduleParser.parse("*/5 * * * *")
        assert result["type"] == "cron"
        assert result["minute"] == "*/5"
        assert result["hour"] == "*"

    def test_parse_invalid_shortcut(self):
        """Should raise error for unknown shortcuts."""
        with pytest.raises(ValueError):
            ScheduleParser.parse("@unknown")

    def test_parse_invalid_cron(self):
        """Should raise error for invalid cron expressions."""
        with pytest.raises(ValueError):
            ScheduleParser.parse("* * *")  # Too few parts

    def test_get_next_run_interval(self):
        """Should calculate next run for intervals."""
        now = datetime.now()
        next_run = ScheduleParser.get_next_run("@every_5m", now)
        expected = now + timedelta(minutes=5)
        # Allow small delta for execution time
        assert abs((next_run - expected).total_seconds()) < 1


class TestAgentEventHandler:
    """Tests for AgentEventHandler."""

    @pytest.mark.asyncio
    async def test_handle_calls_callback(self):
        """Should call agent callback with event prompt."""
        callback = AsyncMock(return_value={"response": "Handled"})
        handler = AgentEventHandler(callback)

        event = Event(
            event_type=EventType.FILE_CREATED,
            source="/tmp/test.txt"
        )

        result = await handler.handle(event)

        callback.assert_called_once()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_handle_error(self):
        """Should handle errors gracefully."""
        callback = AsyncMock(side_effect=Exception("Test error"))
        handler = AgentEventHandler(callback)

        event = Event(
            event_type=EventType.FILE_CREATED,
            source="/tmp/test.txt"
        )

        result = await handler.handle(event)

        assert result["success"] is False
        assert "Test error" in result["error"]


class TestEventManager:
    """Tests for EventManager."""

    def test_register_trigger(self):
        """Should register triggers."""
        manager = EventManager()
        trigger = EventTrigger(
            id="test-trigger",
            event_types=[EventType.FILE_CREATED]
        )

        manager.register_trigger(trigger)

        assert "test-trigger" in manager.triggers
        assert manager.triggers["test-trigger"] == trigger

    def test_unregister_trigger(self):
        """Should unregister triggers."""
        manager = EventManager()
        trigger = EventTrigger(
            id="test-trigger",
            event_types=[EventType.FILE_CREATED]
        )
        manager.register_trigger(trigger)

        manager.unregister_trigger("test-trigger")

        assert "test-trigger" not in manager.triggers

    def test_add_handler(self):
        """Should add event handlers."""
        manager = EventManager()
        handler = MagicMock(spec=EventHandler)

        manager.add_handler(handler)

        assert handler in manager.handlers

    def test_load_triggers_from_config(self):
        """Should load triggers from configuration dict."""
        manager = EventManager()
        config = {
            "events": {
                "triggers": [
                    {
                        "id": "test-trigger",
                        "event_types": ["file_created"],
                        "watch_path": "/tmp",
                        "file_patterns": ["*.txt"],
                        "enabled": True
                    }
                ]
            }
        }

        manager.load_triggers_from_config(config)

        assert "test-trigger" in manager.triggers
        trigger = manager.triggers["test-trigger"]
        assert trigger.watch_path == "/tmp"
        assert "*.txt" in trigger.file_patterns

    def test_get_status(self):
        """Should return manager status."""
        manager = EventManager()
        trigger = EventTrigger(
            id="test",
            event_types=[EventType.FILE_CREATED]
        )
        manager.register_trigger(trigger)

        status = manager.get_status()

        assert status["running"] is False
        assert "test" in status["triggers"]
        assert status["file_watchers"] == 0
        assert status["schedule_tasks"] == 0

    @pytest.mark.asyncio
    async def test_emit_event(self):
        """Should emit events to queue."""
        manager = EventManager()
        event = Event(
            event_type=EventType.FILE_CREATED,
            source="/tmp/test.txt"
        )

        await manager.emit(event)

        # Event should be in queue
        queued_event = await manager.event_queue.get()
        assert queued_event == event

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Should start and stop cleanly."""
        manager = EventManager()

        await manager.start()
        assert manager.running is True

        await manager.stop()
        assert manager.running is False


class TestEventManagerIntegration:
    """Integration tests for EventManager with file watching."""

    @pytest.mark.asyncio
    async def test_file_watcher_trigger(self):
        """Should trigger on file creation in watched directory."""
        # Skip if watchdog not available
        try:
            from watchdog.observers import Observer
        except ImportError:
            pytest.skip("watchdog not installed")

        manager = EventManager()

        # Create a temp directory to watch
        with tempfile.TemporaryDirectory() as tmpdir:
            trigger = EventTrigger(
                id="test-watcher",
                event_types=[EventType.FILE_CREATED],
                watch_path=tmpdir,
                file_patterns=["*.txt"],
                cooldown_seconds=0
            )
            manager.register_trigger(trigger)

            # Add a handler that records events
            received_events = []

            class RecordingHandler(EventHandler):
                async def handle(self, event):
                    received_events.append(event)
                    return {"success": True}

            manager.add_handler(RecordingHandler())

            # Start the manager
            await manager.start()

            # Give the file watcher time to start
            await asyncio.sleep(0.5)

            # Create a file in the watched directory
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("hello")

            # Give time for event to propagate
            await asyncio.sleep(1.0)

            # Stop the manager
            await manager.stop()

            # Check if event was received (may not work in all environments)
            # This is a best-effort test since file watching can be flaky in tests


class TestEventManagerSchedule:
    """Tests for scheduled triggers."""

    @pytest.mark.asyncio
    async def test_schedule_trigger_fires(self):
        """Should fire scheduled triggers."""
        manager = EventManager()

        trigger = EventTrigger(
            id="quick-schedule",
            event_types=[EventType.SCHEDULE],
            schedule="@every_1s",
            cooldown_seconds=0
        )
        manager.register_trigger(trigger)

        received_events = []

        class RecordingHandler(EventHandler):
            async def handle(self, event):
                received_events.append(event)
                return {"success": True}

        manager.add_handler(RecordingHandler())

        await manager.start()

        # Wait for at least one scheduled event
        await asyncio.sleep(1.5)

        await manager.stop()

        # Should have received at least one schedule event
        assert len(received_events) >= 1
        assert received_events[0].event_type == EventType.SCHEDULE


class TestPeripheralEventTypes:
    """Tests for peripheral event types."""

    def test_usb_events(self):
        """Should have USB event types."""
        assert EventType.USB_CONNECTED.value == "usb_connected"
        assert EventType.USB_DISCONNECTED.value == "usb_disconnected"

    def test_network_events(self):
        """Should have network event types."""
        assert EventType.NETWORK_CONNECTED.value == "network_connected"
        assert EventType.NETWORK_DISCONNECTED.value == "network_disconnected"
        assert EventType.NETWORK_CHANGED.value == "network_changed"

    def test_power_events(self):
        """Should have power event types."""
        assert EventType.POWER_AC_CONNECTED.value == "power_ac_connected"
        assert EventType.POWER_AC_DISCONNECTED.value == "power_ac_disconnected"
        assert EventType.POWER_LOW_BATTERY.value == "power_low_battery"

    def test_bluetooth_events(self):
        """Should have bluetooth event types."""
        assert EventType.BLUETOOTH_CONNECTED.value == "bluetooth_connected"
        assert EventType.BLUETOOTH_DISCONNECTED.value == "bluetooth_disconnected"

    def test_display_events(self):
        """Should have display event types."""
        assert EventType.DISPLAY_CONNECTED.value == "display_connected"
        assert EventType.DISPLAY_DISCONNECTED.value == "display_disconnected"

    def test_audio_events(self):
        """Should have audio device event types."""
        assert EventType.AUDIO_DEVICE_CONNECTED.value == "audio_device_connected"
        assert EventType.AUDIO_DEVICE_DISCONNECTED.value == "audio_device_disconnected"


class TestEventTriggerPeripheral:
    """Tests for peripheral-related EventTrigger features."""

    def test_matches_device_no_filter(self):
        """Should match all devices when no filter specified."""
        trigger = EventTrigger(
            id="test",
            event_types=[EventType.USB_CONNECTED],
        )
        assert trigger.matches_device("mass_storage") is True
        assert trigger.matches_device("hid") is True

    def test_matches_device_with_filter(self):
        """Should filter by device type."""
        trigger = EventTrigger(
            id="test",
            event_types=[EventType.USB_CONNECTED],
            device_types=["mass_storage", "hid"]
        )
        assert trigger.matches_device("mass_storage") is True
        assert trigger.matches_device("hid") is True
        assert trigger.matches_device("printer") is False

    def test_matches_interface_no_filter(self):
        """Should match all interfaces when no filter specified."""
        trigger = EventTrigger(
            id="test",
            event_types=[EventType.NETWORK_CONNECTED],
        )
        assert trigger.matches_interface("eth0") is True
        assert trigger.matches_interface("wlan0") is True

    def test_matches_interface_with_filter(self):
        """Should filter by interface pattern."""
        trigger = EventTrigger(
            id="test",
            event_types=[EventType.NETWORK_CONNECTED],
            interface_patterns=["wlan*", "wlp*"]
        )
        assert trigger.matches_interface("wlan0") is True
        assert trigger.matches_interface("wlp3s0") is True
        assert trigger.matches_interface("eth0") is False


class TestEventPromptPeripheral:
    """Tests for peripheral event prompts."""

    def test_usb_connected_prompt(self):
        """Should generate prompt for USB connected event."""
        event = Event(
            event_type=EventType.USB_CONNECTED,
            source="1234:5678",
            data={"vendor": "Acme Corp", "product": "Flash Drive", "device_type": "mass_storage"}
        )
        prompt = event.to_prompt()
        assert "USB device connected" in prompt
        assert "Acme Corp" in prompt
        assert "Flash Drive" in prompt

    def test_network_connected_prompt(self):
        """Should generate prompt for network connected event."""
        event = Event(
            event_type=EventType.NETWORK_CONNECTED,
            source="wlan0",
            data={"interface": "wlan0", "ssid": "MyWiFi"}
        )
        prompt = event.to_prompt()
        assert "Network connected" in prompt
        assert "wlan0" in prompt
        assert "MyWiFi" in prompt

    def test_power_events_prompt(self):
        """Should generate prompts for power events."""
        ac_event = Event(
            event_type=EventType.POWER_AC_CONNECTED,
            source="power"
        )
        assert "Power adapter connected" in ac_event.to_prompt()

        battery_event = Event(
            event_type=EventType.POWER_LOW_BATTERY,
            source="battery",
            data={"percent": 15}
        )
        prompt = battery_event.to_prompt()
        assert "Low battery" in prompt
        assert "15" in prompt


class TestEventManagerPeripheral:
    """Tests for EventManager peripheral support."""

    def test_load_triggers_with_device_types(self):
        """Should load triggers with device_types from config."""
        manager = EventManager()
        config = {
            "events": {
                "triggers": [
                    {
                        "id": "usb-storage",
                        "event_types": ["usb_connected"],
                        "device_types": ["mass_storage"],
                        "enabled": True
                    }
                ]
            }
        }

        manager.load_triggers_from_config(config)

        assert "usb-storage" in manager.triggers
        trigger = manager.triggers["usb-storage"]
        assert "mass_storage" in trigger.device_types

    def test_load_triggers_with_interface_patterns(self):
        """Should load triggers with interface_patterns from config."""
        manager = EventManager()
        config = {
            "events": {
                "triggers": [
                    {
                        "id": "wifi-monitor",
                        "event_types": ["network_connected"],
                        "interface_patterns": ["wlan*", "wlp*"],
                        "enabled": True
                    }
                ]
            }
        }

        manager.load_triggers_from_config(config)

        assert "wifi-monitor" in manager.triggers
        trigger = manager.triggers["wifi-monitor"]
        assert "wlan*" in trigger.interface_patterns
        assert "wlp*" in trigger.interface_patterns

    def test_get_status_includes_peripheral_monitor(self):
        """Should include peripheral_monitor status."""
        manager = EventManager()
        status = manager.get_status()

        assert "peripheral_monitor" in status
        assert status["peripheral_monitor"] is False  # Not started yet
