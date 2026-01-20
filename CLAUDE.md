# meetkatOS Development Guide

This file helps Claude (and other AI assistants) understand the project structure and contribute effectively.

## Project Overview

meetkatOS is an AI-first operating system where AI is the primary interface. The system agent has direct access to OS primitives through tools, with user-configurable autonomy levels.

## Directory Structure

```
meetkatos/
├── system-agent/           # Core AI agent
│   ├── agent.py            # Main agent, tools, inference clients
│   ├── policy.py           # Agency policy system
│   └── requirements.txt
├── ai-shell/               # Terminal interface
│   ├── shell.py            # Conversational shell
│   └── requirements.txt
├── config/
│   └── agency.toml         # Agency policy configuration
├── tests/                  # Test suite
│   ├── conftest.py         # Shared fixtures
│   ├── test_policy.py      # Policy module tests
│   ├── test_agent.py       # Agent module tests
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

### AI Shell (`ai-shell/shell.py`)

- **AgentClient**: HTTP/Unix socket client to agent
- **Builtin commands**: /help, /profile, /policy, /reset, /exit

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
| `AI_INFERENCE_BACKEND` | "anthropic" or "openai" | auto-detect |
| `AI_MODEL_NAME` | Model to use | claude-sonnet-4-20250514 |
| `AI_RUNTIME_URL` | OpenAI-compatible URL | (none) |
| `AI_DEV_MODE` | Enable HTTP server mode | (disabled) |
| `AI_POLICY_CONFIG` | Path to agency.toml | (auto-detect) |

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
