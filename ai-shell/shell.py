#!/usr/bin/env python3
"""
AI Shell - The primary user interface for AI-OS

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

    async def get_events(self) -> dict:
        return await self._request("GET", "/events")

    async def enable_trigger(self, trigger_id: str) -> dict:
        return await self._request("POST", "/events/trigger/enable", {"id": trigger_id})

    async def disable_trigger(self, trigger_id: str) -> dict:
        return await self._request("POST", "/events/trigger/disable", {"id": trigger_id})

    async def get_audit_summary(self, hours: int = 24) -> dict:
        return await self._request("GET", f"/audit/summary?hours={hours}")

    async def get_audit_entries(self, count: int = 50, action_filter: str = None) -> dict:
        url = f"/audit/entries?count={count}"
        if action_filter:
            url += f"&filter={action_filter}"
        return await self._request("GET", url)


# =============================================================================
# Shell Commands
# =============================================================================

BUILTIN_COMMANDS = {
    "/help": "Show this help message",
    "/clear": "Clear the screen",
    "/reset": "Reset conversation (new session)",
    "/profile": "Show or change agency profile",
    "/policy": "Check policy for an action",
    "/events": "Manage event triggers",
    "/audit": "View audit log",
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

    elif cmd == "/events":
        await handle_events_command(args, client)
        return True

    elif cmd == "/audit":
        await handle_audit_command(args, client)
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


async def handle_events_command(args: list, client: AgentClient):
    """Handle /events subcommands."""

    if not args:
        # Show status
        try:
            status = await client.get_events()
            console.print("\n[bold]Event System Status[/bold]\n")
            console.print(f"  Running: {'[green]yes[/green]' if status.get('running') else '[yellow]no[/yellow]'}")
            console.print(f"  File watchers: {status.get('file_watchers', 0)}")
            console.print(f"  Schedule tasks: {status.get('schedule_tasks', 0)}")
            console.print(f"  Queue size: {status.get('queue_size', 0)}")

            triggers = status.get("triggers", {})
            if triggers:
                console.print("\n[bold]Triggers[/bold]\n")
                for tid, tinfo in triggers.items():
                    enabled = "[green]✓[/green]" if tinfo.get("enabled") else "[dim]✗[/dim]"
                    event_types = ", ".join(tinfo.get("event_types", []))
                    console.print(f"  {enabled} [cyan]{tid:25}[/cyan] {event_types}")
                    if tinfo.get("watch_path"):
                        console.print(f"      Watch: {tinfo['watch_path']}")
                    if tinfo.get("schedule"):
                        console.print(f"      Schedule: {tinfo['schedule']}")
                    if tinfo.get("last_triggered"):
                        console.print(f"      [dim]Last: {tinfo['last_triggered']}[/dim]")
            console.print()
        except Exception as e:
            console.print(f"[red]Error getting event status: {e}[/red]")
        return

    subcmd = args[0].lower()

    if subcmd == "enable" and len(args) > 1:
        trigger_id = args[1]
        try:
            result = await client.enable_trigger(trigger_id)
            if result.get("success"):
                console.print(f"[green]Enabled trigger: {trigger_id}[/green]")
            else:
                console.print(f"[red]Failed to enable: {result.get('error', 'unknown')}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    elif subcmd == "disable" and len(args) > 1:
        trigger_id = args[1]
        try:
            result = await client.disable_trigger(trigger_id)
            if result.get("success"):
                console.print(f"[yellow]Disabled trigger: {trigger_id}[/yellow]")
            else:
                console.print(f"[red]Failed to disable: {result.get('error', 'unknown')}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    elif subcmd == "help":
        console.print("\n[bold]Event Commands[/bold]")
        console.print("  [cyan]/events[/cyan]              Show event system status")
        console.print("  [cyan]/events enable ID[/cyan]    Enable a trigger")
        console.print("  [cyan]/events disable ID[/cyan]   Disable a trigger")
        console.print()

    else:
        console.print("[yellow]Usage: /events [enable|disable ID][/yellow]")
        console.print("Run [cyan]/events help[/cyan] for more info.")


async def handle_audit_command(args: list, client: AgentClient):
    """Handle /audit subcommands."""

    if not args:
        # Show summary
        try:
            summary = await client.get_audit_summary(24)
            console.print("\n[bold]Audit Summary (last 24h)[/bold]\n")
            console.print(f"  Total entries:    {summary.get('total_entries', 0)}")
            console.print(f"  Events triggered: {summary.get('events_triggered', 0)}")
            console.print(f"  User denials:     {summary.get('user_denials', 0)}")
            console.print(f"  Errors:           {summary.get('errors', 0)}")

            tools = summary.get("tools_used", {})
            if tools:
                console.print("\n[bold]Tools Used[/bold]")
                for tool, count in sorted(tools.items(), key=lambda x: -x[1])[:5]:
                    console.print(f"  {tool:20} {count}")

            actions = summary.get("actions", {})
            if actions:
                console.print("\n[bold]Actions[/bold]")
                for action, count in sorted(actions.items(), key=lambda x: -x[1])[:5]:
                    console.print(f"  {action:25} {count}")
            console.print()
        except Exception as e:
            console.print(f"[red]Error getting audit summary: {e}[/red]")
        return

    subcmd = args[0].lower()

    if subcmd == "log" or subcmd == "entries":
        # Parse arguments: /audit log [count] [--filter=type] [--tool=name]
        count = 20
        action_filter = None

        for arg in args[1:]:
            if arg.startswith("--filter="):
                action_filter = arg.split("=", 1)[1]
            elif arg.startswith("--tool="):
                action_filter = f"tool:{arg.split('=', 1)[1]}"
            elif arg.startswith("--type="):
                action_filter = arg.split("=", 1)[1]
            elif arg.isdigit():
                count = int(arg)

        try:
            result = await client.get_audit_entries(count, action_filter)
            entries = result.get("entries", [])

            if not entries:
                console.print("[dim]No audit entries found.[/dim]")
                return

            console.print(f"\n[bold]Recent Audit Entries ({len(entries)})[/bold]\n")
            for entry in entries:
                ts = entry.get("timestamp", "")[:19]  # Trim microseconds
                action = entry.get("action", "unknown")

                # Color code by action type
                if "failed" in action or "denied" in action:
                    color = "red"
                elif "completed" in action or "confirmed" in action:
                    color = "green"
                elif "invoked" in action or "triggered" in action:
                    color = "yellow"
                else:
                    color = "dim"

                line = f"[dim]{ts}[/dim] [{color}]{action:20}[/{color}]"

                if entry.get("tool_name"):
                    line += f" [cyan]{entry['tool_name']}[/cyan]"
                if entry.get("trigger_id"):
                    line += f" [magenta]{entry['trigger_id']}[/magenta]"
                if entry.get("domain"):
                    line += f" {entry['domain']}.{entry.get('operation', '')}"
                if entry.get("error"):
                    line += f" [red]{entry['error'][:50]}[/red]"

                console.print(line)
            console.print()
        except Exception as e:
            console.print(f"[red]Error getting audit entries: {e}[/red]")

    elif subcmd == "help":
        console.print("\n[bold]Audit Commands[/bold]")
        console.print("  [cyan]/audit[/cyan]                    Show 24h summary")
        console.print("  [cyan]/audit log[/cyan]                Show recent entries")
        console.print("  [cyan]/audit log N[/cyan]              Show N recent entries")
        console.print("  [cyan]/audit log --type=TYPE[/cyan]    Filter by action type")
        console.print("  [cyan]/audit log --tool=NAME[/cyan]    Filter by tool name")
        console.print()
        console.print("[bold]Action types:[/bold] tool_invoked, tool_completed, tool_failed,")
        console.print("               policy_check, user_confirmed, user_denied,")
        console.print("               event_triggered, event_handled")
        console.print()
        console.print("[bold]Log file location:[/bold] ~/.ai-os/audit/audit.jsonl")
        console.print()

    else:
        console.print("[yellow]Usage: /audit [log [N]][/yellow]")
        console.print("Run [cyan]/audit help[/cyan] for more info.")


# =============================================================================
# Display Helpers
# =============================================================================


def print_welcome():
    console.print(Panel.fit(
        "[bold green]AI-OS Shell[/bold green]\n"
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
