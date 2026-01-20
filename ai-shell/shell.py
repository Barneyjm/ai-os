#!/usr/bin/env python3
"""
AI Shell - The primary user interface for MeerkatOS

A conversational terminal interface that connects to the system agent.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
from aiohttp import UnixConnector
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# =============================================================================
# Configuration
# =============================================================================

AGENT_SOCKET = os.environ.get("AI_AGENT_SOCKET", "/run/system-agent.sock")
AGENT_URL = os.environ.get("AI_AGENT_URL", "")  # HTTP URL for dev mode
HISTORY_FILE = Path.home() / ".ai_shell_history"
SESSION_ID = f"shell-{os.getpid()}"

STYLE = Style.from_dict({
    'prompt': '#00aa00 bold',
    'path': '#888888',
})

console = Console()

# =============================================================================
# Agent Client
# =============================================================================


class AgentClient:
    """Client for communicating with the system agent."""

    def __init__(self, socket_path: str = None, url: str = None):
        self.socket_path = socket_path
        self.url = url
        self.session_id = SESSION_ID
        self.use_http = bool(url)

    async def _request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make a request to the agent."""
        if self.use_http:
            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(f"{self.url}{endpoint}") as resp:
                        return await resp.json()
                else:
                    async with session.post(f"{self.url}{endpoint}", json=data or {}) as resp:
                        return await resp.json()
        else:
            connector = UnixConnector(path=self.socket_path)
            async with aiohttp.ClientSession(connector=connector) as session:
                if method == "GET":
                    async with session.get(f"http://localhost{endpoint}") as resp:
                        return await resp.json()
                else:
                    async with session.post(f"http://localhost{endpoint}", json=data or {}) as resp:
                        return await resp.json()

    async def send_message(self, message: str) -> dict:
        return await self._request("POST", "/chat", {
            "session_id": self.session_id,
            "message": message
        })

    async def get_profiles(self) -> dict:
        return await self._request("GET", "/profiles")

    async def set_profile(self, profile_name: str) -> dict:
        return await self._request("POST", "/profiles/set", {"profile": profile_name})

    async def check_policy(self, domain: str, operation: str, target: str = "") -> dict:
        return await self._request("POST", "/policy/check", {
            "domain": domain,
            "operation": operation,
            "target": target,
            "description": f"{operation} {target}"
        })

    async def get_health(self) -> dict:
        return await self._request("GET", "/health")


# =============================================================================
# Shell Commands
# =============================================================================

BUILTIN_COMMANDS = {
    "/help": "Show this help message",
    "/clear": "Clear the screen",
    "/reset": "Reset conversation (new session)",
    "/profile": "Show or change agency profile",
    "/policy": "Check policy for an action",
    "/exit": "Exit the shell",
}


async def handle_builtin(command: str, client: AgentClient) -> bool:
    """Handle built-in commands. Returns True if command was handled."""

    parts = command.strip().split()
    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []

    if cmd == "/help":
        console.print("\n[bold]AI Shell Commands[/bold]\n")
        for c, desc in BUILTIN_COMMANDS.items():
            console.print(f"  [cyan]{c:12}[/cyan] {desc}")
        console.print("\n[bold]Profile Commands[/bold]")
        console.print("  [cyan]/profile[/cyan]          Show current profile")
        console.print("  [cyan]/profile list[/cyan]     List available profiles")
        console.print("  [cyan]/profile set NAME[/cyan] Switch to profile")
        console.print("\nType anything else to chat with the AI agent.\n")
        return True

    elif cmd == "/clear":
        console.clear()
        return True

    elif cmd == "/exit":
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)

    elif cmd == "/reset":
        global SESSION_ID
        SESSION_ID = f"shell-{os.getpid()}-{datetime.now().timestamp()}"
        client.session_id = SESSION_ID
        console.print("[dim]Session reset.[/dim]")
        return True

    elif cmd == "/profile":
        await handle_profile_command(args, client)
        return True

    elif cmd == "/policy":
        if len(args) >= 2:
            await handle_policy_check(args, client)
        else:
            console.print("[yellow]Usage: /policy DOMAIN OPERATION [TARGET][/yellow]")
            console.print("Example: /policy filesystem write /etc/nginx.conf")
        return True

    return False


async def handle_profile_command(args: list, client: AgentClient):
    """Handle /profile subcommands."""

    if not args:
        profiles = await client.get_profiles()
        active = profiles.get("active", "default")
        console.print(f"[bold]Active profile:[/bold] {active or 'default'}")
        return

    subcmd = args[0].lower()

    if subcmd == "list":
        profiles = await client.get_profiles()
        console.print("\n[bold]Available Profiles[/bold]\n")
        for name, desc in profiles.get("profiles", {}).items():
            marker = " [green]✓[/green]" if name == profiles.get("active") else ""
            console.print(f"  [cyan]{name:15}[/cyan] {desc}{marker}")
        console.print()

    elif subcmd == "set" and len(args) > 1:
        profile_name = args[1]
        result = await client.set_profile(profile_name)
        if result.get("success"):
            console.print(f"[green]Switched to profile: {profile_name}[/green]")
        else:
            console.print(f"[red]Error: {result.get('error')}[/red]")

    else:
        console.print("[yellow]Usage: /profile [list|set NAME][/yellow]")


async def handle_policy_check(args: list, client: AgentClient):
    """Check what policy applies to an action."""
    domain = args[0]
    operation = args[1]
    target = args[2] if len(args) > 2 else ""

    result = await client.check_policy(domain, operation, target)

    level_colors = {
        "observe": "dim",
        "suggest": "blue",
        "confirm": "yellow",
        "auto": "green",
        "autonomous": "cyan"
    }
    level = result.get("level", "unknown")
    color = level_colors.get(level, "white")

    console.print(f"\n[bold]Policy for {domain}.{operation}[/bold]")
    console.print(f"  Level:  [{color}]{level}[/{color}]")
    console.print(f"  Source: {result.get('source', 'unknown')}")
    console.print(f"  Reason: {result.get('reason', 'N/A')}")
    console.print()


# =============================================================================
# Display Helpers
# =============================================================================


def print_welcome():
    console.print(Panel.fit(
        "[bold green]MeerkatOS Shell[/bold green]\n"
        "[dim]Type naturally to interact with your system.[/dim]\n"
        "[dim]Type /help for commands.[/dim]",
        border_style="green"
    ))
    console.print()


def print_response(response: dict):
    """Print agent response with markdown rendering and notifications."""

    notifications = response.get("notifications", [])
    for notif in notifications:
        console.print(f"[dim]ℹ {notif.get('message', '')}[/dim]")

    text = response.get("response", "")
    if not text:
        return

    if any(marker in text for marker in ['```', '**', '# ', '- ', '1. ']):
        md = Markdown(text)
        console.print(md)
    else:
        console.print(text)

    suggestions = response.get("policy_suggestions", [])
    if suggestions:
        console.print()
        console.print("[yellow]💡 Policy suggestions available.[/yellow]")

    console.print()


def get_prompt():
    cwd = os.getcwd()
    home = str(Path.home())
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]

    return HTML(f'<path>{cwd}</path> <prompt>❯</prompt> ')


# =============================================================================
# Main Loop
# =============================================================================


async def main():
    print_welcome()

    # Use HTTP in dev mode, Unix socket in production
    if AGENT_URL:
        client = AgentClient(url=AGENT_URL)
        console.print(f"[dim]Connecting to: {AGENT_URL}[/dim]\n")
    else:
        client = AgentClient(socket_path=AGENT_SOCKET)

    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        style=STYLE
    )

    # Show current profile on startup
    try:
        profiles = await client.get_profiles()
        active = profiles.get("active")
        if active:
            console.print(f"[dim]Active profile: {active}[/dim]\n")
    except:
        pass

    while True:
        try:
            user_input = await session.prompt_async(get_prompt)

            if not user_input.strip():
                continue

            if user_input.startswith('/'):
                if await handle_builtin(user_input, client):
                    continue

            with console.status("[bold green]Thinking...", spinner="dots"):
                try:
                    response = await client.send_message(user_input)
                    print_response(response)
                except aiohttp.ClientError as e:
                    console.print(f"[red]Error connecting to agent: {e}[/red]")
                    console.print("[dim]Is the system-agent running?[/dim]")
                    console.print("[dim]For dev mode, set AI_AGENT_URL=http://127.0.0.1:8000[/dim]")

        except KeyboardInterrupt:
            console.print("\n[dim]Use /exit to quit[/dim]")
            continue

        except EOFError:
            break

    console.print("[dim]Goodbye![/dim]")


if __name__ == "__main__":
    asyncio.run(main())
