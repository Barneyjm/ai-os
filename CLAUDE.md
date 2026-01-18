# AI-OS Development Guide

This file helps Claude (and other AI assistants) understand the project structure and contribute effectively.

## Project Overview

AI-OS is an AI-first operating system where AI is the primary interface. The system agent has direct access to OS primitives through tools, with user-configurable autonomy levels.

## Directory Structure

```
ai-os/
├── system-agent/           # Core AI agent
│   ├── agent.py            # Main agent, tools, inference clients
│   ├── policy.py           # Agency policy system
│   ├── events.py           # Event system for proactive triggers
│   └── requirements.txt
├── ai-shell/               # Terminal interface
│   ├── shell.py            # Conversational shell
│   └── requirements.txt
├── config/
│   └── agency.toml         # Agency policy + event trigger configuration
├── tests/                  # Test suite
│   ├── conftest.py         # Shared fixtures
│   ├── test_policy.py      # Policy module tests
│   ├── test_agent.py       # Agent module tests
│   ├── test_events.py      # Event system tests
│   └── test_shell.py       # Shell module tests
├── services/               # runit service definitions
├── docs/                   # Documentation
├── buildroot-external/     # Linux distro build config
├── .github/workflows/      # CI/CD
│   └── ci.yml
├── pyproject.toml          # Project config & pytest settings
└── CLAUDE.md               # This file
```

## Key Components

### Agency Policy System (`system-agent/policy.py`)

Controls AI autonomy across domains:
- **AgencyLevel**: OBSERVE → SUGGEST → CONFIRM → AUTO → AUTONOMOUS
- **Action**: Represents an action the AI wants to take
- **PolicyDecision**: Result of policy check with level, reason, source
- **AgencyPolicy**: Main policy engine with profile support

### System Agent (`system-agent/agent.py`)

- **AgentConfig**: Configuration from environment variables
- **Tool/ToolRegistry**: OS tool definitions and handlers
- **InferenceClient**: OpenAI-compatible API client
- **AnthropicInferenceClient**: Anthropic Messages API client
- **SystemAgent**: Main agent orchestrating tools and policy

### Event System (`system-agent/events.py`)

Provides proactive agency through automatic triggers:
- **EventType**: FILE_CREATED, FILE_MODIFIED, SCHEDULE, LOW_DISK_SPACE, etc.
- **Event**: Represents something that happened with timestamp and data
- **EventTrigger**: Defines when to fire (file patterns, schedules, thresholds)
- **EventManager**: Coordinates file watchers, schedulers, and system monitors
- **AgentEventHandler**: Routes events to the SystemAgent for processing

### AI Shell (`ai-shell/shell.py`)

- **AgentClient**: HTTP/Unix socket client to agent
- **Builtin commands**: /help, /profile, /policy, /events, /reset, /exit

## Development Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_policy.py -v
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | (none) |
| `ANTHROPIC_BASE_URL` | Anthropic API base URL | https://api.anthropic.com |
| `OPENAI_API_KEY` | OpenAI-compatible API key (Fireworks, OpenAI, etc.) | (none) |
| `AI_INFERENCE_BACKEND` | "anthropic" or "openai" | auto-detect |
| `AI_MODEL_NAME` | Model to use | claude-sonnet-4-20250514 |
| `AI_RUNTIME_URL` | OpenAI-compatible URL | (none) |
| `AI_DEV_MODE` | Enable HTTP server mode | (disabled) |
| `AI_POLICY_CONFIG` | Path to agency.toml | (auto-detect) |

### Backend Configuration Examples

**Anthropic (Claude)**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export AI_MODEL_NAME="claude-sonnet-4-20250514"
```

**Fireworks AI**:
```bash
export OPENAI_API_KEY="fw_..."
export AI_RUNTIME_URL="https://api.fireworks.ai/inference/v1"
export AI_MODEL_NAME="accounts/fireworks/models/llama-v3p1-70b-instruct"
```

**OpenAI**:
```bash
export OPENAI_API_KEY="sk-..."
export AI_RUNTIME_URL="https://api.openai.com/v1"
export AI_MODEL_NAME="gpt-4o"
```

**Local llama.cpp** (no auth):
```bash
export AI_RUNTIME_URL="http://localhost:8080"
export AI_MODEL_NAME="llama-3-8b"
```

## Testing Guidelines

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_agent.py

# Specific test class
pytest tests/test_agent.py::TestAnthropicInferenceClient

# Specific test
pytest tests/test_agent.py::TestAnthropicInferenceClient::test_convert_messages_simple -v
```

### Writing Tests

1. **Location**: Place tests in `tests/test_<module>.py`
2. **Fixtures**: Add shared fixtures to `tests/conftest.py`
3. **Async tests**: Use `@pytest.mark.asyncio` decorator
4. **Naming**: `test_<function>_<scenario>`

Example test structure:
```python
class TestComponentName:
    """Tests for ComponentName."""

    def test_basic_functionality(self):
        """Should do the basic thing."""
        # Arrange
        component = Component()

        # Act
        result = component.do_thing()

        # Assert
        assert result == expected

    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Should handle async operation."""
        result = await component.async_method()
        assert result["success"] is True
```

### Test Coverage Areas

When adding features, ensure tests cover:
- [ ] Happy path (normal operation)
- [ ] Error cases (invalid input, failures)
- [ ] Edge cases (empty input, boundary values)
- [ ] Integration with other components

## Adding New Features

### Adding a New Tool

1. Add tool definition in `ToolRegistry._register_builtin_tools()`:
```python
self.register(Tool(
    name="tool_name",
    description="What it does",
    parameters={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "..."}
        },
        "required": ["param"]
    },
    handler=self._tool_handler,
    domain="domain_name",
    operation="operation_name"
))
```

2. Add handler method:
```python
async def _tool_handler(self, param: str) -> dict:
    try:
        # Implementation
        return {"success": True, "result": ...}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

3. Add tests in `tests/test_agent.py`:
```python
@pytest.mark.asyncio
async def test_tool_handler_success(self, registry):
    result = await registry._tool_handler(param="value")
    assert result["success"] is True
```

### Adding a New Inference Backend

1. Create client class in `agent.py`:
```python
class NewBackendClient:
    def __init__(self, config: AgentConfig):
        self.config = config

    async def complete(self, messages, tools=None, **kwargs) -> dict:
        # Must return OpenAI-compatible response format
        return {
            "choices": [{
                "message": {"role": "assistant", "content": "..."},
                "finish_reason": "stop"
            }]
        }
```

2. Update `create_inference_client()` factory
3. Add config options to `AgentConfig`
4. Add tests for message conversion and response handling

### Adding a New Agency Domain

1. Add domain config in `config/agency.toml`:
```toml
[domains.new_domain]
operation1 = "auto"
operation2 = "confirm"
sensitive_items = ["item1", "item2"]
```

2. Add tests in `tests/test_policy.py`:
```python
def test_get_policy_new_domain(self, temp_config_file):
    policy = AgencyPolicy(config_path=temp_config_file)
    action = Action(
        domain="new_domain",
        operation="operation1",
        target="target",
        description="Test"
    )
    decision = policy.get_policy(action)
    assert decision.level == AgencyLevel.AUTO
```

### Adding Event Triggers

Event triggers enable proactive AI behavior. Add them to `config/agency.toml`:

**File Watcher Trigger**:
```toml
[[events.triggers]]
id = "downloads-organizer"
enabled = true
event_types = ["file_created"]
watch_path = "~/Downloads"
file_patterns = ["*.pdf", "*.zip"]
ignore_patterns = [".*", "*.tmp"]
cooldown_seconds = 30
prompt = "A new file was downloaded. Suggest where to organize it."
```

**Scheduled Trigger**:
```toml
[[events.triggers]]
id = "daily-summary"
enabled = true
event_types = ["schedule"]
schedule = "@daily"  # or "0 9 * * *" for 9am, "@every_5m", etc.
prompt = "Provide a daily system health summary."
```

**System Monitor Trigger**:
```toml
[[events.triggers]]
id = "disk-warning"
enabled = true
event_types = ["low_disk_space"]
watch_path = "/"
threshold = 90.0  # Trigger when >90% used
cooldown_seconds = 3600
prompt = "Disk space is low. Suggest cleanup actions."
```

**Peripheral Event Triggers**:
```toml
# USB device connected (filter by device type)
[[events.triggers]]
id = "usb-storage-handler"
enabled = true
event_types = ["usb_connected"]
device_types = ["mass_storage"]  # Filter: mass_storage, hid, audio, video, etc.
cooldown_seconds = 10
prompt = "A USB storage device was connected. List its contents."

# Network connection changes (filter by interface)
[[events.triggers]]
id = "wifi-monitor"
enabled = true
event_types = ["network_connected", "network_disconnected"]
interface_patterns = ["wlan*", "wlp*"]  # Only WiFi interfaces
cooldown_seconds = 30
prompt = "WiFi connection changed. Check connectivity status."

# Power/battery events
[[events.triggers]]
id = "low-battery-handler"
enabled = true
event_types = ["power_low_battery"]
threshold = 20.0  # Battery percentage
cooldown_seconds = 300
prompt = "Battery is low. Suggest power-saving actions."
```

**Peripheral Event Types**:
- USB: `usb_connected`, `usb_disconnected`
- Network: `network_connected`, `network_disconnected`, `network_changed`
- Power: `power_ac_connected`, `power_ac_disconnected`, `power_low_battery`
- Bluetooth: `bluetooth_connected`, `bluetooth_disconnected`
- Display: `display_connected`, `display_disconnected`
- Audio: `audio_device_connected`, `audio_device_disconnected`

**Schedule Syntax**:
- `@daily`, `@hourly`, `@weekly`, `@monthly`
- `@every_5m`, `@every_1h`, `@every_30s`
- Cron: `*/5 * * * *` (every 5 minutes)

**Shell Commands**:
- `/events` - Show event system status and triggers
- `/events enable ID` - Enable a trigger
- `/events disable ID` - Disable a trigger

## CI/CD Pipeline

GitHub Actions runs on push/PR to main:

1. **test**: pytest on Python 3.10, 3.11, 3.12
2. **lint**: ruff linting and format check
3. **type-check**: mypy type checking
4. **integration**: CLI and initialization tests
5. **security**: bandit security scan

## Common Tasks

### Update dependencies
```bash
pip install -e ".[dev]" --upgrade
```

### Format code
```bash
pip install ruff
ruff format system-agent/ ai-shell/ tests/
```

### Type check
```bash
pip install mypy
mypy system-agent/ --ignore-missing-imports
```

### Security scan
```bash
pip install bandit
bandit -r system-agent/ ai-shell/ -ll
```

## Architecture Notes

### Message Flow
```
User Input → AI Shell → System Agent → Inference Client → LLM
                ↓
         Policy Engine (check autonomy level)
                ↓
         Tool Execution (if approved)
                ↓
         Response → AI Shell → User
```

### Event System Flow
```
                    ┌─────────────────┐
                    │  Event Manager  │
                    └────────┬────────┘
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
   File Watcher        Scheduler        System Monitor
   (watchdog)       (cron-like)       (CPU/mem/disk)
         │                  │                  │
         └──────────────────┼──────────────────┘
                            ▼
                      Event Queue
                            ▼
                   AgentEventHandler
                            ▼
                     SystemAgent
                   (process_message)
                            ▼
                    LLM + Tool Use
```

### Tool Schema Formats

**OpenAI format** (for llama.cpp, OpenAI, etc.):
```json
{
  "type": "function",
  "function": {
    "name": "tool_name",
    "description": "...",
    "parameters": {...}
  }
}
```

**Anthropic format**:
```json
{
  "name": "tool_name",
  "description": "...",
  "input_schema": {...}
}
```

The `ToolRegistry.get_tool_schemas(format=)` handles conversion.

### Response Normalization

All inference clients return OpenAI-compatible responses:
```python
{
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "text response",
            "tool_calls": [...]  # optional
        },
        "finish_reason": "stop" | "tool_calls" | "length"
    }]
}
```

This allows the SystemAgent to work with any backend uniformly.
