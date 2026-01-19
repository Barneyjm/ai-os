#!/usr/bin/env python3
"""
AI-OS System Agent

The core agent that mediates between user intent and OS primitives.
This is the "brain" of the AI-first operating system.
"""

import asyncio
import json
import os
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from datetime import datetime

import aiohttp
from aiohttp import web, UnixConnector

from policy import AgencyPolicy, AgencyLevel, Action, PolicyDecision
from events import EventManager, EventType, Event, EventTrigger, AgentEventHandler
from audit import get_audit_logger, AuditAction
from knowledge import get_knowledge

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AgentConfig:
    runtime_socket: str = "/run/ai-runtime.sock"
    runtime_url: str = ""  # HTTP URL for development mode
    agent_socket: str = "/run/system-agent.sock"
    model_name: str = "llama-3-8b"
    max_context: int = 8192
    consent_timeout: int = 30
    log_level: str = "INFO"
    policy_config: Optional[str] = None
    # OpenAI-compatible API settings (Fireworks, OpenAI, etc.)
    openai_api_key: str = ""
    # Anthropic-specific settings
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    inference_backend: str = "openai"  # "openai" or "anthropic"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables."""
        # Determine backend: explicit setting > auto-detect from keys
        explicit_backend = os.environ.get("AI_INFERENCE_BACKEND", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")

        if explicit_backend:
            backend = explicit_backend
        elif anthropic_key:
            backend = "anthropic"
        elif openai_key:
            backend = "openai"
        else:
            backend = "openai"  # Default to OpenAI-compatible (local llama.cpp)

        return cls(
            runtime_socket=os.environ.get("AI_RUNTIME_SOCKET", "/run/ai-runtime.sock"),
            runtime_url=os.environ.get("AI_RUNTIME_URL", ""),
            agent_socket=os.environ.get("AI_AGENT_SOCKET", "/run/system-agent.sock"),
            model_name=os.environ.get("AI_MODEL_NAME", "claude-sonnet-4-20250514"),
            policy_config=os.environ.get("AI_POLICY_CONFIG"),
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            anthropic_base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            inference_backend=backend,
        )


# =============================================================================
# Tool Definitions
# =============================================================================


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable
    domain: str
    operation: str

    def to_action(self, **kwargs) -> Action:
        """Convert tool call to Action for policy checking."""
        target = kwargs.get("path") or kwargs.get("pid") or kwargs.get("service") or kwargs.get("command", "")
        return Action(
            domain=self.domain,
            operation=self.operation,
            target=str(target),
            description=f"{self.name}: {target}",
            metadata=kwargs
        )


class ToolRegistry:
    """Registry of all tools available to the AI agent."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._register_builtin_tools()

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tool_schemas(self, format: str = "openai") -> list[dict]:
        """Return tool schemas in specified format.

        Args:
            format: "openai" for OpenAI/llama.cpp format, "anthropic" for Anthropic format
        """
        if format == "anthropic":
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters
                }
                for t in self.tools.values()
            ]
        else:
            # OpenAI format (default)
            return [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                }
                for t in self.tools.values()
            ]

    def _register_builtin_tools(self):
        """Register core OS tools."""

        self.register(Tool(
            name="read_file",
            description="Read the contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"}
                },
                "required": ["path"]
            },
            handler=self._read_file,
            domain="filesystem",
            operation="read"
        ))

        self.register(Tool(
            name="write_file",
            description="Write content to a file (creates if doesn't exist)",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            },
            handler=self._write_file,
            domain="filesystem",
            operation="write_user_files"
        ))

        self.register(Tool(
            name="list_directory",
            description="List contents of a directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to directory"},
                    "recursive": {"type": "boolean", "description": "List recursively", "default": False}
                },
                "required": ["path"]
            },
            handler=self._list_directory,
            domain="filesystem",
            operation="list"
        ))

        self.register(Tool(
            name="find_files",
            description="Search for files matching a pattern",
            parameters={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to search in"},
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.txt')"},
                    "content_match": {"type": "string", "description": "Optional: search file contents"}
                },
                "required": ["directory", "pattern"]
            },
            handler=self._find_files,
            domain="filesystem",
            operation="search"
        ))

        self.register(Tool(
            name="run_command",
            description="Execute a shell command and return output",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                    "working_dir": {"type": "string", "description": "Working directory"}
                },
                "required": ["command"]
            },
            handler=self._run_command,
            domain="processes",
            operation="run_command"
        ))

        self.register(Tool(
            name="list_processes",
            description="List running processes",
            parameters={
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "description": "Filter by process name"}
                }
            },
            handler=self._list_processes,
            domain="processes",
            operation="list"
        ))

        self.register(Tool(
            name="kill_process",
            description="Terminate a process by PID",
            parameters={
                "type": "object",
                "properties": {
                    "pid": {"type": "integer", "description": "Process ID"},
                    "signal": {"type": "string", "description": "Signal to send", "default": "TERM"}
                },
                "required": ["pid"]
            },
            handler=self._kill_process,
            domain="processes",
            operation="kill_user_processes"
        ))

        self.register(Tool(
            name="service_status",
            description="Get status of a system service",
            parameters={
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"}
                },
                "required": ["service"]
            },
            handler=self._service_status,
            domain="services",
            operation="status"
        ))

        self.register(Tool(
            name="service_control",
            description="Start, stop, or restart a service",
            parameters={
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "action": {"type": "string", "enum": ["start", "stop", "restart"]}
                },
                "required": ["service", "action"]
            },
            handler=self._service_control,
            domain="services",
            operation="restart"
        ))

        self.register(Tool(
            name="package_search",
            description="Search for available packages",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Package name or keyword"}
                },
                "required": ["query"]
            },
            handler=self._package_search,
            domain="packages",
            operation="search"
        ))

        self.register(Tool(
            name="package_install",
            description="Install a package",
            parameters={
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "Package name"}
                },
                "required": ["package"]
            },
            handler=self._package_install,
            domain="packages",
            operation="install"
        ))

        self.register(Tool(
            name="system_info",
            description="Get system information (CPU, memory, disk, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["all", "cpu", "memory", "disk", "network"],
                        "default": "all"
                    }
                }
            },
            handler=self._system_info,
            domain="system",
            operation="resources"
        ))

        self.register(Tool(
            name="system_power",
            description="Shutdown, reboot, or suspend the system",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["shutdown", "reboot", "suspend"]}
                },
                "required": ["action"]
            },
            handler=self._system_power,
            domain="system",
            operation="shutdown"
        ))

    # --- Tool Implementations ---

    async def _read_file(self, path: str) -> dict:
        try:
            content = Path(path).read_text()
            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _write_file(self, path: str, content: str) -> dict:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content)
            return {"success": True, "path": path, "bytes_written": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _list_directory(self, path: str, recursive: bool = False) -> dict:
        try:
            p = Path(path)
            if recursive:
                items = [str(f.relative_to(p)) for f in p.rglob("*")]
            else:
                items = [
                    {"name": f.name, "type": "dir" if f.is_dir() else "file", "size": f.stat().st_size if f.is_file() else None}
                    for f in p.iterdir()
                ]
            return {"success": True, "items": items}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _find_files(self, directory: str, pattern: str, content_match: str = None) -> dict:
        try:
            matches = list(Path(directory).rglob(pattern))
            results = []
            for m in matches[:100]:
                entry = {"path": str(m)}
                if content_match and m.is_file():
                    try:
                        if content_match in m.read_text():
                            results.append(entry)
                    except:
                        pass
                else:
                    results.append(entry)
            return {"success": True, "matches": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_command(self, command: str, timeout: int = 30, working_dir: str = None) -> dict:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "success": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode()
            }
        except asyncio.TimeoutError:
            return {"success": False, "error": "Command timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _list_processes(self, filter: str = None) -> dict:
        result = await self._run_command("ps aux")
        if result["success"]:
            lines = result["stdout"].strip().split("\n")
            if filter:
                lines = [l for l in lines if filter.lower() in l.lower()]
            return {"success": True, "processes": lines}
        return result

    async def _kill_process(self, pid: int, signal: str = "TERM") -> dict:
        return await self._run_command(f"kill -{signal} {pid}")

    async def _service_status(self, service: str) -> dict:
        # Try systemctl first, fall back to sv
        result = await self._run_command(f"systemctl status {service} 2>/dev/null || sv status {service}")
        return result

    async def _service_control(self, service: str, action: str) -> dict:
        result = await self._run_command(f"systemctl {action} {service} 2>/dev/null || sv {action} {service}")
        return result

    async def _package_search(self, query: str) -> dict:
        # Try apt first, fall back to apk
        result = await self._run_command(f"apt-cache search {query} 2>/dev/null || apk search {query}")
        return result

    async def _package_install(self, package: str) -> dict:
        result = await self._run_command(f"apt-get install -y {package} 2>/dev/null || apk add {package}")
        return result

    async def _system_info(self, category: str = "all") -> dict:
        info = {}
        if category in ("all", "cpu"):
            info["cpu"] = (await self._run_command("cat /proc/cpuinfo | grep 'model name' | head -1"))["stdout"]
            info["load"] = (await self._run_command("cat /proc/loadavg"))["stdout"]
        if category in ("all", "memory"):
            info["memory"] = (await self._run_command("free -h"))["stdout"]
        if category in ("all", "disk"):
            info["disk"] = (await self._run_command("df -h"))["stdout"]
        if category in ("all", "network"):
            info["network"] = (await self._run_command("ip addr 2>/dev/null || ifconfig"))["stdout"]
        return {"success": True, "info": info}

    async def _system_power(self, action: str) -> dict:
        commands = {
            "shutdown": "poweroff",
            "reboot": "reboot",
            "suspend": "systemctl suspend 2>/dev/null || echo mem > /sys/power/state"
        }
        return await self._run_command(commands[action])


# =============================================================================
# Inference Client
# =============================================================================


class InferenceClient:
    """Client for communicating with OpenAI-compatible inference runtimes.

    Supports local llama.cpp (no auth), Fireworks AI, OpenAI, and other
    OpenAI-compatible APIs that use Bearer token authentication.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.use_http = bool(config.runtime_url)
        self.logger = logging.getLogger("openai-client")

    def _get_headers(self) -> dict:
        """Build request headers, including auth if API key is set."""
        headers = {"Content-Type": "application/json"}
        if self.config.openai_api_key:
            headers["Authorization"] = f"Bearer {self.config.openai_api_key}"
        return headers

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> dict:
        """Send a completion request to the inference runtime."""

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = self._get_headers()

        if self.use_http:
            # HTTP mode (cloud APIs or local dev server)
            url = f"{self.config.runtime_url}/v1/chat/completions"
            self.logger.debug(f"Sending request to {url}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.logger.error(f"API error {resp.status}: {error_text}")
                        raise Exception(f"API error {resp.status}: {error_text}")
                    return await resp.json()
        else:
            # Unix socket mode (production)
            connector = UnixConnector(path=self.config.runtime_socket)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    "http://localhost/v1/chat/completions",
                    json=payload,
                    headers=headers
                ) as resp:
                    return await resp.json()


class AnthropicInferenceClient:
    """Client for communicating with the Anthropic Messages API."""

    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger("anthropic-client")

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> dict:
        """Send a completion request to the Anthropic Messages API.

        Converts between OpenAI-style messages and Anthropic format,
        then converts the response back to OpenAI format for compatibility.
        """
        # Convert messages to Anthropic format
        anthropic_messages, system_prompt = self._convert_messages_to_anthropic(messages)

        payload = {
            "model": self.config.model_name,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = {"type": "auto"}

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
        }

        url = f"{self.config.anthropic_base_url}/v1/messages"

        self.logger.debug(f"Sending request to {url}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    self.logger.error(f"Anthropic API error {resp.status}: {error_text}")
                    raise Exception(f"Anthropic API error {resp.status}: {error_text}")

                response = await resp.json()
                # Convert response to OpenAI format for compatibility
                return self._convert_response_to_openai(response)

    def _convert_messages_to_anthropic(self, messages: list[dict]) -> tuple[list[dict], str]:
        """Convert OpenAI-style messages to Anthropic format.

        Returns (messages, system_prompt) tuple.
        """
        anthropic_messages = []
        system_prompt = ""

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                # Anthropic uses a separate system parameter
                system_prompt = content
            elif role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                # Check if this message has tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Convert tool calls to Anthropic format
                    content_blocks = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in msg["tool_calls"]:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                        })
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
                else:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content
                    })
            elif role == "tool":
                # Tool results in Anthropic format go as user messages with tool_result content
                tool_call_id = msg.get("tool_call_id", "")
                tool_content = msg.get("content", "")

                # Find if the last message is a user message with tool_result blocks
                # If so, append to it; otherwise create a new one
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    last_content = anthropic_messages[-1]["content"]
                    if isinstance(last_content, list):
                        last_content.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": tool_content
                        })
                        continue

                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": tool_content
                    }]
                })

        return anthropic_messages, system_prompt

    def _convert_response_to_openai(self, anthropic_response: dict) -> dict:
        """Convert Anthropic response to OpenAI format for compatibility."""
        content_blocks = anthropic_response.get("content", [])
        stop_reason = anthropic_response.get("stop_reason", "end_turn")

        # Extract text content and tool uses
        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if block["type"] == "text":
                text_content += block.get("text", "")
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"])
                    }
                })

        # Build OpenAI-compatible response
        message = {
            "role": "assistant",
            "content": text_content if text_content else None,
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        # Map stop reasons
        finish_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }

        return {
            "id": anthropic_response.get("id", ""),
            "object": "chat.completion",
            "model": anthropic_response.get("model", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason_map.get(stop_reason, "stop")
            }],
            "usage": anthropic_response.get("usage", {})
        }


def create_inference_client(config: AgentConfig):
    """Factory function to create the appropriate inference client."""
    if config.inference_backend == "anthropic":
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic backend")
        return AnthropicInferenceClient(config)
    else:
        return InferenceClient(config)


# =============================================================================
# System Agent
# =============================================================================


class SystemAgent:
    """The core AI agent that processes user requests and orchestrates tools."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools = ToolRegistry()
        self.inference = create_inference_client(config)
        self.conversations: dict[str, list] = {}
        self.logger = logging.getLogger("system-agent")

        # Determine tool schema format based on backend
        self.tool_schema_format = "anthropic" if config.inference_backend == "anthropic" else "openai"

        policy_path = Path(config.policy_config) if config.policy_config else None
        self.policy = AgencyPolicy(policy_path)

        self.pending_notifications: dict[str, list] = {}

        # Event system for proactive triggers
        self.event_manager = EventManager()
        self._setup_event_system()

        # Audit logging
        self.audit = get_audit_logger()
        self.audit.log_agent_lifecycle(AuditAction.AGENT_STARTED, {
            "backend": config.inference_backend,
            "model": config.model_name
        })

        self.logger.info(f"Using {config.inference_backend} backend with model {config.model_name}")

    def _setup_event_system(self):
        """Initialize the event system with triggers from config."""
        # Load triggers from policy config if available
        if self.policy.config:
            self.event_manager.load_triggers_from_config(self.policy.config)

        # Add handler that routes events to the agent
        handler = AgentEventHandler(self._handle_event)
        self.event_manager.add_handler(handler)

    async def _handle_event(self, prompt: str, event: Event) -> dict:
        """Handle an event by processing it as an agent message."""
        # Use a dedicated session for event-triggered actions
        session_id = f"event:{event.trigger_id or 'unknown'}"

        # Get the trigger to check auto_approve setting
        trigger = self.event_manager.triggers.get(event.trigger_id)

        async def event_confirm(action: Action, decision: PolicyDecision) -> bool:
            # If trigger has auto_approve, skip confirmation
            if trigger and trigger.auto_approve:
                return True
            # Otherwise, log and deny (events can't get interactive confirmation)
            self.logger.info(f"Event action requires confirmation (denied): {action.description}")
            return False

        def event_notify(action: Action, msg: str):
            self.logger.info(f"Event notification: {msg}")
            self.pending_notifications.setdefault("events", []).append({
                "action": action.description,
                "message": msg,
                "event_type": event.event_type.value,
                "timestamp": datetime.now().isoformat()
            })

        result = await self.process_message(
            session_id=session_id,
            user_message=prompt,
            confirm_callback=event_confirm,
            notify_callback=event_notify
        )

        return result

    async def start_event_system(self):
        """Start the event monitoring system."""
        await self.event_manager.start()
        self.logger.info("Event system started")

    async def stop_event_system(self):
        """Stop the event monitoring system."""
        await self.event_manager.stop()
        self.logger.info("Event system stopped")

    def get_event_status(self) -> dict:
        """Get the current status of the event system."""
        return self.event_manager.get_status()

    def register_event_trigger(self, trigger: EventTrigger):
        """Register a new event trigger at runtime."""
        self.event_manager.register_trigger(trigger)

    def unregister_event_trigger(self, trigger_id: str):
        """Remove an event trigger."""
        self.event_manager.unregister_trigger(trigger_id)

    async def emit_event(self, event: Event):
        """Manually emit an event to trigger handlers."""
        await self.event_manager.emit(event)

    def _get_system_prompt(self) -> str:
        profile_info = ""
        if self.policy.active_profile:
            profile_info = f"\nCurrent agency profile: {self.policy.active_profile}"

        # Get situational knowledge (can be customized via ~/.ai-os/knowledge.md)
        knowledge = get_knowledge(concise=False)

        return f"""You are the System Agent for an AI-first operating system. You have direct access to OS primitives through tools.

Your role is to:
1. Understand user intent and translate it into appropriate system operations
2. Use tools to accomplish tasks - prefer tools over explaining how to do things manually
3. Respect the user's agency policy - some actions require confirmation, others are automatic
4. Provide clear, concise feedback about what you're doing

Guidelines:
- For file operations, always use absolute paths
- When running commands that might produce a lot of output, consider piping to head/tail
- If a task requires multiple steps, execute them in sequence and report progress
- If something fails, diagnose the issue and suggest alternatives
- Be proactive - if you notice something relevant while completing a task, mention it
- If an action is denied by policy, explain what happened and suggest alternatives

You have access to these capabilities: filesystem operations, process management, service control, package management, and system information. Use them freely to help the user.{profile_info}

{knowledge}"""

    async def process_message(
        self,
        session_id: str,
        user_message: str,
        confirm_callback: Callable[[Action, PolicyDecision], bool] = None,
        notify_callback: Callable[[Action, str], None] = None
    ) -> dict:
        """Process a user message and return the agent's response."""

        if session_id not in self.conversations:
            self.conversations[session_id] = [
                {"role": "system", "content": self._get_system_prompt()}
            ]

        if session_id not in self.pending_notifications:
            self.pending_notifications[session_id] = []

        messages = self.conversations[session_id]
        messages.append({"role": "user", "content": user_message})

        if not confirm_callback:
            async def default_confirm(action, decision):
                return True
            confirm_callback = default_confirm

        if not notify_callback:
            def default_notify(action, msg):
                self.pending_notifications[session_id].append({
                    "action": action.description,
                    "message": msg,
                    "timestamp": datetime.now().isoformat()
                })
            notify_callback = default_notify

        while True:
            response = await self.inference.complete(
                messages=messages,
                tools=self.tools.get_tool_schemas(format=self.tool_schema_format)
            )

            assistant_message = response["choices"][0]["message"]
            messages.append(assistant_message)

            if "tool_calls" not in assistant_message or not assistant_message["tool_calls"]:
                notifications = self.pending_notifications.get(session_id, [])
                self.pending_notifications[session_id] = []

                return {
                    "response": assistant_message.get("content", ""),
                    "notifications": notifications,
                    "policy_suggestions": self.policy.get_pending_suggestions()
                }

            for tool_call in assistant_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])

                self.logger.info(f"Tool call: {tool_name} with args: {tool_args}")

                tool = self.tools.tools.get(tool_name)
                if not tool:
                    result = {"error": f"Unknown tool: {tool_name}"}
                    self.audit.log_tool_failed(session_id, tool_name, f"Unknown tool: {tool_name}")
                else:
                    action = tool.to_action(**tool_args)
                    decision = self.policy.get_policy(action)

                    self.logger.info(
                        f"Policy decision: {decision.level} for {action.policy_key} "
                        f"(from {decision.source})"
                    )

                    # Audit the policy check
                    self.audit.log_policy_check(
                        session_id, action.domain, action.operation, action.target,
                        str(decision.level), decision.source
                    )

                    match decision.level:
                        case AgencyLevel.OBSERVE:
                            result = {
                                "policy": "observe",
                                "message": f"Action logged but not executed per policy: {action.description}"
                            }

                        case AgencyLevel.SUGGEST:
                            notify_callback(action, f"Suggestion: {action.description}")
                            result = {
                                "policy": "suggest",
                                "message": f"Suggested to user: {action.description}"
                            }

                        case AgencyLevel.CONFIRM:
                            import time
                            start = time.time()
                            approved = await confirm_callback(action, decision)
                            response_time = int((time.time() - start) * 1000)

                            self.policy.record_response(action, decision, approved, response_time)

                            # Audit user decision
                            self.audit.log_user_decision(
                                session_id, approved, action.domain, action.operation,
                                action.target, response_time
                            )

                            if approved:
                                result = await self._execute_tool_with_audit(
                                    session_id, tool, tool_name, tool_args, action
                                )
                            else:
                                result = {
                                    "policy": "denied",
                                    "message": f"User denied: {action.description}"
                                }

                        case AgencyLevel.AUTO:
                            result = await self._execute_tool_with_audit(
                                session_id, tool, tool_name, tool_args, action
                            )
                            notify_callback(action, f"Completed: {action.description}")

                        case AgencyLevel.AUTONOMOUS:
                            result = await self._execute_tool_with_audit(
                                session_id, tool, tool_name, tool_args, action
                            )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })

        if len(messages) > 50:
            messages = messages[:1] + messages[-40:]
            self.conversations[session_id] = messages

    async def _execute_tool_with_audit(
        self,
        session_id: str,
        tool,
        tool_name: str,
        tool_args: dict,
        action
    ) -> dict:
        """Execute a tool with full audit logging."""
        import time

        # Log invocation
        self.audit.log_tool_invoked(
            session_id, tool_name, tool_args,
            action.domain, action.operation, action.target
        )

        start = time.time()
        try:
            result = await tool.handler(**tool_args)
            duration_ms = int((time.time() - start) * 1000)

            success = result.get("success", False) if isinstance(result, dict) else False
            self.audit.log_tool_completed(session_id, tool_name, result, duration_ms, success)

            return result

        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            self.audit.log_tool_failed(session_id, tool_name, str(e), duration_ms)
            return {"success": False, "error": str(e)}

    def set_profile(self, profile_name: str) -> bool:
        """Change the active agency profile."""
        old_profile = self.policy.active_profile
        success = self.policy.set_profile(profile_name)
        if success:
            # Audit the profile change
            self.audit.log_profile_changed("system", old_profile or "default", profile_name)
            for session_id, messages in self.conversations.items():
                if messages and messages[0]["role"] == "system":
                    messages[0]["content"] = self._get_system_prompt()
        return success

    def get_policy_summary(self) -> dict:
        """Get a summary of current policy settings."""
        return {
            "active_profile": self.policy.active_profile,
            "available_profiles": self.policy.list_profiles(),
            "pending_suggestions": self.policy.get_pending_suggestions()
        }

    def get_audit_summary(self, hours: int = 24) -> dict:
        """Get a summary of audit activity."""
        return self.audit.get_summary(hours)

    def get_recent_audit_entries(self, count: int = 50, action_filter: list = None) -> list:
        """Get recent audit log entries."""
        entries = self.audit.get_recent_entries(count, action_filter)
        return [entry.to_dict() for entry in entries]


# =============================================================================
# IPC Server
# =============================================================================


async def create_ipc_server(agent: SystemAgent, socket_path: str):
    """Create Unix socket server for IPC with shells and other clients."""

    async def handle_chat(request: web.Request) -> web.Response:
        data = await request.json()
        session_id = data.get("session_id", "default")
        message = data.get("message", "")

        async def confirm(action: Action, decision: PolicyDecision) -> bool:
            return True

        def notify(action: Action, msg: str):
            pass

        result = await agent.process_message(session_id, message, confirm, notify)
        return web.json_response(result)

    async def handle_profile_list(request: web.Request) -> web.Response:
        profiles = agent.policy.list_profiles()
        active = agent.policy.active_profile
        return web.json_response({"profiles": profiles, "active": active})

    async def handle_profile_set(request: web.Request) -> web.Response:
        data = await request.json()
        profile_name = data.get("profile")

        if not profile_name:
            return web.json_response({"error": "No profile specified"}, status=400)

        success = agent.set_profile(profile_name)
        if success:
            return web.json_response({"success": True, "active": profile_name})
        else:
            return web.json_response({"error": f"Unknown profile: {profile_name}"}, status=404)

    async def handle_policy_check(request: web.Request) -> web.Response:
        data = await request.json()
        action = Action(
            domain=data.get("domain", ""),
            operation=data.get("operation", ""),
            target=data.get("target", ""),
            description=data.get("description", "")
        )
        decision = agent.policy.get_policy(action)
        return web.json_response({
            "level": str(decision.level),
            "reason": decision.reason,
            "source": decision.source,
            "requires_confirmation": decision.requires_confirmation
        })

    async def handle_policy_suggestions(request: web.Request) -> web.Response:
        suggestions = agent.policy.get_pending_suggestions()
        return web.json_response({"suggestions": suggestions})

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy",
            "active_profile": agent.policy.active_profile,
            "active_sessions": len(agent.conversations),
            "events_running": agent.event_manager.running
        })

    async def handle_events_status(request: web.Request) -> web.Response:
        return web.json_response(agent.get_event_status())

    async def handle_events_enable(request: web.Request) -> web.Response:
        data = await request.json()
        trigger_id = data.get("id")
        if not trigger_id:
            return web.json_response({"success": False, "error": "No trigger ID specified"}, status=400)

        if trigger_id not in agent.event_manager.triggers:
            return web.json_response({"success": False, "error": f"Unknown trigger: {trigger_id}"}, status=404)

        agent.event_manager.triggers[trigger_id].enabled = True
        return web.json_response({"success": True, "trigger": trigger_id, "enabled": True})

    async def handle_events_disable(request: web.Request) -> web.Response:
        data = await request.json()
        trigger_id = data.get("id")
        if not trigger_id:
            return web.json_response({"success": False, "error": "No trigger ID specified"}, status=400)

        if trigger_id not in agent.event_manager.triggers:
            return web.json_response({"success": False, "error": f"Unknown trigger: {trigger_id}"}, status=404)

        agent.event_manager.triggers[trigger_id].enabled = False
        return web.json_response({"success": True, "trigger": trigger_id, "enabled": False})

    async def handle_audit_summary(request: web.Request) -> web.Response:
        hours = int(request.query.get("hours", "24"))
        summary = agent.get_audit_summary(hours)
        return web.json_response(summary)

    async def handle_audit_entries(request: web.Request) -> web.Response:
        count = int(request.query.get("count", "50"))
        action_filter = request.query.get("filter", "").split(",") if request.query.get("filter") else None
        if action_filter:
            action_filter = [a.strip() for a in action_filter if a.strip()]
        entries = agent.get_recent_audit_entries(count, action_filter)
        return web.json_response({"entries": entries})

    app = web.Application()
    app.router.add_post("/chat", handle_chat)
    app.router.add_get("/profiles", handle_profile_list)
    app.router.add_post("/profiles/set", handle_profile_set)
    app.router.add_post("/policy/check", handle_policy_check)
    app.router.add_get("/policy/suggestions", handle_policy_suggestions)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/events", handle_events_status)
    app.router.add_post("/events/trigger/enable", handle_events_enable)
    app.router.add_post("/events/trigger/disable", handle_events_disable)
    app.router.add_get("/audit/summary", handle_audit_summary)
    app.router.add_get("/audit/entries", handle_audit_entries)

    if os.path.exists(socket_path):
        os.unlink(socket_path)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.UnixSite(runner, socket_path)
    await site.start()

    os.chmod(socket_path, 0o666)

    return runner


# =============================================================================
# HTTP Server (Development Mode)
# =============================================================================


async def create_http_server(agent: SystemAgent, host: str = "127.0.0.1", port: int = 8000):
    """Create HTTP server for development/testing."""

    async def handle_chat(request: web.Request) -> web.Response:
        data = await request.json()
        session_id = data.get("session_id", "default")
        message = data.get("message", "")

        async def confirm(action: Action, decision: PolicyDecision) -> bool:
            return True

        result = await agent.process_message(session_id, message, confirm)
        return web.json_response(result)

    async def handle_profile_list(request: web.Request) -> web.Response:
        profiles = agent.policy.list_profiles()
        active = agent.policy.active_profile
        return web.json_response({"profiles": profiles, "active": active})

    async def handle_profile_set(request: web.Request) -> web.Response:
        data = await request.json()
        profile_name = data.get("profile")
        success = agent.set_profile(profile_name)
        if success:
            return web.json_response({"success": True, "active": profile_name})
        return web.json_response({"error": f"Unknown profile: {profile_name}"}, status=404)

    async def handle_policy_check(request: web.Request) -> web.Response:
        data = await request.json()
        action = Action(
            domain=data.get("domain", ""),
            operation=data.get("operation", ""),
            target=data.get("target", ""),
            description=data.get("description", "")
        )
        decision = agent.policy.get_policy(action)
        return web.json_response({
            "level": str(decision.level),
            "reason": decision.reason,
            "source": decision.source
        })

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({"status": "healthy"})

    app = web.Application()
    app.router.add_post("/chat", handle_chat)
    app.router.add_get("/profiles", handle_profile_list)
    app.router.add_post("/profiles/set", handle_profile_set)
    app.router.add_post("/policy/check", handle_policy_check)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    return runner


# =============================================================================
# Main
# =============================================================================


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s'
    )
    logger = logging.getLogger("main")

    config = AgentConfig.from_env()
    agent = SystemAgent(config)

    logger.info("Starting System Agent...")

    # Use HTTP server in development mode, Unix socket in production
    use_http = bool(os.environ.get("AI_DEV_MODE") or config.runtime_url)

    if use_http:
        port = int(os.environ.get("AI_AGENT_PORT", "8000"))
        runner = await create_http_server(agent, port=port)
        logger.info(f"HTTP server listening on http://127.0.0.1:{port}")
    else:
        runner = await create_ipc_server(agent, config.agent_socket)
        logger.info(f"IPC server listening on {config.agent_socket}")

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
