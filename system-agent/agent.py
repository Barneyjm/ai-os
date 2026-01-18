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

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables."""
        return cls(
            runtime_socket=os.environ.get("AI_RUNTIME_SOCKET", "/run/ai-runtime.sock"),
            runtime_url=os.environ.get("AI_RUNTIME_URL", ""),
            agent_socket=os.environ.get("AI_AGENT_SOCKET", "/run/system-agent.sock"),
            model_name=os.environ.get("AI_MODEL_NAME", "llama-3-8b"),
            policy_config=os.environ.get("AI_POLICY_CONFIG"),
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

    def get_tool_schemas(self) -> list[dict]:
        """Return tool schemas in OpenAI function-calling format."""
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
    """Client for communicating with the inference runtime."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.use_http = bool(config.runtime_url)

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

        if self.use_http:
            # HTTP mode (development)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.runtime_url}/v1/chat/completions",
                    json=payload
                ) as resp:
                    return await resp.json()
        else:
            # Unix socket mode (production)
            connector = UnixConnector(path=self.config.runtime_socket)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    "http://localhost/v1/chat/completions",
                    json=payload
                ) as resp:
                    return await resp.json()


# =============================================================================
# System Agent
# =============================================================================


class SystemAgent:
    """The core AI agent that processes user requests and orchestrates tools."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools = ToolRegistry()
        self.inference = InferenceClient(config)
        self.conversations: dict[str, list] = {}
        self.logger = logging.getLogger("system-agent")

        policy_path = Path(config.policy_config) if config.policy_config else None
        self.policy = AgencyPolicy(policy_path)

        self.pending_notifications: dict[str, list] = {}

    def _get_system_prompt(self) -> str:
        profile_info = ""
        if self.policy.active_profile:
            profile_info = f"\nCurrent agency profile: {self.policy.active_profile}"

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

You have access to these capabilities: filesystem operations, process management, service control, package management, and system information. Use them freely to help the user.{profile_info}"""

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
                tools=self.tools.get_tool_schemas()
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
                else:
                    action = tool.to_action(**tool_args)
                    decision = self.policy.get_policy(action)

                    self.logger.info(
                        f"Policy decision: {decision.level} for {action.policy_key} "
                        f"(from {decision.source})"
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

                            if approved:
                                result = await tool.handler(**tool_args)
                            else:
                                result = {
                                    "policy": "denied",
                                    "message": f"User denied: {action.description}"
                                }

                        case AgencyLevel.AUTO:
                            result = await tool.handler(**tool_args)
                            notify_callback(action, f"Completed: {action.description}")

                        case AgencyLevel.AUTONOMOUS:
                            result = await tool.handler(**tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })

        if len(messages) > 50:
            messages = messages[:1] + messages[-40:]
            self.conversations[session_id] = messages

    def set_profile(self, profile_name: str) -> bool:
        """Change the active agency profile."""
        success = self.policy.set_profile(profile_name)
        if success:
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
            "active_sessions": len(agent.conversations)
        })

    app = web.Application()
    app.router.add_post("/chat", handle_chat)
    app.router.add_get("/profiles", handle_profile_list)
    app.router.add_post("/profiles/set", handle_profile_set)
    app.router.add_post("/policy/check", handle_policy_check)
    app.router.add_get("/policy/suggestions", handle_policy_suggestions)
    app.router.add_get("/health", handle_health)

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
