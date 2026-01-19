"""
Situational Intelligence Knowledge Base for AI-OS System Agent

This module provides contextual knowledge that helps the AI agent
make better decisions about system administration tasks.

Knowledge can be customized by creating ~/.ai-os/knowledge.md or
setting AI_KNOWLEDGE_PATH environment variable.
"""

import os
from pathlib import Path

# =============================================================================
# System Administration Knowledge
# =============================================================================

SYSADMIN_KNOWLEDGE = """
## Command Reference by Task

### User Management
- Create user: `useradd -m -s /bin/bash USERNAME` (use -m for home dir)
- Delete user: `userdel -r USERNAME` (-r removes home dir)
- Add to group: `usermod -aG GROUP USERNAME`
- Set password: `passwd USERNAME` (interactive) or `chpasswd` (scripted)
- List users: `cat /etc/passwd` or `getent passwd`
- SSH key: Add to `~USER/.ssh/authorized_keys` (create dir with 700, file with 600)

### Service Management
- systemd: `systemctl {start|stop|restart|status|enable|disable} SERVICE`
- Check logs: `journalctl -u SERVICE -f` (follow) or `-n 50` (last 50 lines)
- List failed: `systemctl --failed`
- Reload config: `systemctl daemon-reload` (after editing unit files)

### Process Troubleshooting
- Find by name: `pgrep -f PATTERN` or `ps aux | grep PATTERN`
- Find by port: `lsof -i :PORT` or `ss -tlnp | grep PORT`
- Kill gracefully: `kill PID` (SIGTERM), then `kill -9 PID` (SIGKILL) if needed
- Top memory: `ps aux --sort=-%mem | head -20`
- Top CPU: `ps aux --sort=-%cpu | head -20`
- Process tree: `pstree -p PID`

### Disk & Storage
- Usage summary: `df -h`
- Directory size: `du -sh /path` or `du -h --max-depth=1 /path`
- Find large files: `find /path -type f -size +100M -exec ls -lh {} \\;`
- Find large dirs: `du -h /path | sort -rh | head -20`
- Clear package cache:
  - Debian/Ubuntu: `apt clean`
  - RHEL/Fedora: `dnf clean all`
  - Arch: `pacman -Sc`
- Clear journal: `journalctl --vacuum-size=100M`

### Memory Management
- Current usage: `free -h`
- Clear caches (safe): `sync; echo 3 > /proc/sys/vm/drop_caches`
- Find memory hogs: `ps aux --sort=-%mem | head -10`
- Check OOM kills: `dmesg | grep -i "out of memory"` or `journalctl -k | grep -i oom`
- Swap usage by process: `for f in /proc/*/status; do awk '/VmSwap/{print $2}' $f 2>/dev/null; done | sort -n | tail`

### Network Diagnostics
- Interfaces: `ip addr` or `ip a`
- Routes: `ip route`
- DNS test: `dig DOMAIN` or `nslookup DOMAIN`
- Port check: `nc -zv HOST PORT` or `curl -I http://HOST:PORT`
- Connections: `ss -tuln` (listening) or `ss -tun` (established)
- Firewall status:
  - ufw: `ufw status verbose`
  - firewalld: `firewall-cmd --list-all`
  - iptables: `iptables -L -n -v`

### Log Analysis
- System log: `journalctl -xe` (recent with context)
- Boot log: `journalctl -b` (current boot) or `-b -1` (previous)
- Auth failures: `journalctl -u sshd | grep -i fail`
- Kernel messages: `dmesg -T | tail -50`
- Follow multiple: `tail -f /var/log/syslog /var/log/auth.log`

### Package Management
- Debian/Ubuntu: `apt update && apt upgrade`, `apt install PKG`, `apt search PKG`
- RHEL/Fedora: `dnf upgrade`, `dnf install PKG`, `dnf search PKG`
- Arch: `pacman -Syu`, `pacman -S PKG`, `pacman -Ss PKG`
- Check what provides file: `dpkg -S /path/file` or `rpm -qf /path/file`

### Security Checks
- Failed SSH: `journalctl -u sshd | grep -i "failed\\|invalid"`
- Failed logins: `lastb | head -20`
- Current sessions: `who` or `w`
- Listening services: `ss -tlnp`
- SUID files: `find / -perm -4000 -type f 2>/dev/null`
- World-writable: `find /path -perm -002 -type f 2>/dev/null`

## Troubleshooting Heuristics

### Service Won't Start
1. Check status: `systemctl status SERVICE` - look for error messages
2. Check logs: `journalctl -u SERVICE -n 50`
3. Check config: Most services have a test mode (`nginx -t`, `apache2ctl configtest`)
4. Check dependencies: `systemctl list-dependencies SERVICE`
5. Check resources: Is disk full? Memory exhausted? Port already in use?

### High CPU
1. Identify process: `top` or `ps aux --sort=-%cpu | head`
2. Check if it's expected (compiling, processing, etc.)
3. Check for runaway loops: `strace -p PID` (briefly)
4. Consider: nice/renice, cgroup limits, or restart if it's a known issue

### High Memory
1. Identify hogs: `ps aux --sort=-%mem | head`
2. Check for memory leaks (growing over time)
3. Safe first: Clear caches `echo 3 > /proc/sys/vm/drop_caches`
4. Check swap: `free -h` - high swap with free RAM = something's wrong
5. Consider: Restart leaky service, add swap, adjust OOM priorities

### Disk Full
1. Find where: `df -h` to identify mount point
2. Find what: `du -h --max-depth=1 /path | sort -rh | head`
3. Quick wins:
   - Package cache: `apt clean` / `dnf clean all`
   - Journal: `journalctl --vacuum-size=100M`
   - Temp files: `find /tmp -type f -atime +7 -delete`
   - Old logs: Check /var/log for large rotated logs
4. Find large files: `find / -type f -size +500M 2>/dev/null`

### Network Unreachable
1. Check interface: `ip addr` - is it up? Has IP?
2. Check gateway: `ip route` - is default route set?
3. Check DNS: `cat /etc/resolv.conf`, try `ping 8.8.8.8` vs `ping google.com`
4. Check firewall: Is it blocking? `iptables -L -n`
5. Check service: `ss -tlnp` - is the service actually listening?

### SSH Connection Refused
1. Is sshd running? `systemctl status sshd`
2. Is it listening? `ss -tlnp | grep 22`
3. Firewall blocking? `iptables -L -n | grep 22`
4. Check config: `sshd -t` to test configuration
5. Check auth log: `journalctl -u sshd -n 50`

## Safety Guidelines

### Always Verify Before
- Deleting files: Use `ls` first, especially with wildcards
- Killing processes: Verify PID belongs to expected process
- Modifying config: Back up first or use `cp file file.bak`
- Package removal: Check what depends on it

### Dangerous Commands - Extra Caution
- `rm -rf /` or any recursive delete near root
- `chmod -R` or `chown -R` on system directories
- `dd` - verify source and destination carefully
- `mkfs` - this destroys data
- `iptables -F` - may lock you out

### Best Practices
- Use `--dry-run` when available
- Pipe destructive commands to `less` first: `find ... -delete` → `find ...` first
- For critical services, do restarts during maintenance windows
- Keep a terminal open before changing SSH config
"""

# =============================================================================
# Distro Detection Hints
# =============================================================================

DISTRO_DETECTION = """
## Detecting Distribution

Check for package manager:
- `/usr/bin/apt` → Debian/Ubuntu
- `/usr/bin/dnf` → Fedora/RHEL 8+
- `/usr/bin/yum` → RHEL 7/CentOS 7
- `/usr/bin/pacman` → Arch
- `/usr/bin/apk` → Alpine

Or check: `cat /etc/os-release`

## Distro-Specific Notes

### Debian/Ubuntu
- Services: systemd
- Packages: apt (dpkg underneath)
- Firewall: ufw (iptables underneath)
- Config: /etc/default/ for service defaults

### RHEL/Fedora/CentOS
- Services: systemd
- Packages: dnf (or yum on older)
- Firewall: firewalld (firewall-cmd)
- SELinux: Check `getenforce`, may need `restorecon`

### Arch
- Services: systemd
- Packages: pacman
- Config: Minimal, often need AUR for extras

### Alpine
- Services: OpenRC (`rc-service`, `rc-update`)
- Packages: apk
- Minimal: Many tools need explicit install
"""

# =============================================================================
# Event Response Patterns
# =============================================================================

EVENT_RESPONSE_PATTERNS = """
## Responding to Events

### High Memory Event
1. Run `free -h` to confirm
2. Run `ps aux --sort=-%mem | head -10` to identify consumers
3. Check `dmesg | grep -i oom` for recent OOM kills
4. If a known-leaky service is top consumer, consider restart
5. If system is stable but memory high, clearing cache is safe
6. Report findings and actions taken

### Low Disk Event
1. Run `df -h` to identify which filesystem
2. Run `du -h --max-depth=1 /path | sort -rh | head` to find culprits
3. Safe cleanups: package cache, old journals, temp files
4. Report large files/directories that could be cleaned
5. If critical (<5%), identify what to delete immediately

### Service Failed Event
1. Check `systemctl status SERVICE` for error
2. Check `journalctl -u SERVICE -n 50` for details
3. Common fixes: restart, reload config, check dependencies
4. If config error, identify and explain
5. Restart if appropriate for the service

### USB Device Connected
1. Run `lsblk` to see block devices
2. Check `dmesg | tail -20` for device info
3. If storage device, check if auto-mounted
4. Report device type and suggested actions

### Network Changed
1. Run `ip addr` and `ip route` to see current state
2. Check `cat /etc/resolv.conf` for DNS
3. Test connectivity: `ping -c 1 8.8.8.8`
4. If disconnected, report what's missing
5. If connected, verify internet access works
"""


def get_full_knowledge() -> str:
    """Return the complete knowledge base for the system prompt."""
    return f"""
{SYSADMIN_KNOWLEDGE}

{DISTRO_DETECTION}

{EVENT_RESPONSE_PATTERNS}
"""


def get_concise_knowledge() -> str:
    """Return a condensed version for context-limited models."""
    return """
## Quick Reference

### Common Commands
- Users: useradd -m, userdel -r, usermod -aG
- Services: systemctl {start|stop|restart|status|enable}
- Logs: journalctl -u SERVICE -n 50, journalctl -xe
- Disk: df -h, du -sh /path, find -size +100M
- Memory: free -h, ps aux --sort=-%mem | head
- Network: ip addr, ss -tlnp, curl -I URL
- Processes: ps aux, pgrep, kill PID

### Package Managers
- Debian/Ubuntu: apt
- RHEL/Fedora: dnf
- Arch: pacman
- Alpine: apk

### Troubleshooting Order
1. Check status/logs first
2. Verify config (most have -t test flag)
3. Check resources (disk, memory, ports)
4. Check dependencies
5. Restart as last resort

### Safety
- Verify before delete (ls first)
- Backup before config changes
- Use --dry-run when available
"""


# =============================================================================
# Configurable Knowledge Loading
# =============================================================================

def _get_custom_knowledge_path() -> Path | None:
    """Find custom knowledge file if it exists."""
    # Check environment variable first
    env_path = os.environ.get("AI_KNOWLEDGE_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Check ~/.ai-os/knowledge.md
    home_path = Path.home() / ".ai-os" / "knowledge.md"
    if home_path.exists():
        return home_path

    # Check /etc/ai-os/knowledge.md
    etc_path = Path("/etc/ai-os/knowledge.md")
    if etc_path.exists():
        return etc_path

    return None


def _load_custom_knowledge() -> str | None:
    """Load custom knowledge from user-defined file."""
    path = _get_custom_knowledge_path()
    if path:
        try:
            return path.read_text()
        except Exception:
            return None
    return None


def get_knowledge(concise: bool = False) -> str:
    """
    Get the knowledge base for the system prompt.

    Checks for custom knowledge file in order:
    1. AI_KNOWLEDGE_PATH environment variable
    2. ~/.ai-os/knowledge.md
    3. /etc/ai-os/knowledge.md

    If no custom file found, uses built-in knowledge.

    Args:
        concise: If True, return condensed version (for smaller context models)

    Returns:
        Knowledge base string to include in system prompt
    """
    # Try custom knowledge first
    custom = _load_custom_knowledge()
    if custom:
        return custom

    # Fall back to built-in
    if concise:
        return get_concise_knowledge()
    return get_full_knowledge()
