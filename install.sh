#!/bin/bash
#
# MeerAI Install Script
# Installs MeerAI system agent and shell
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/Barneyjm/ai-os/main/install.sh | bash
#
# Or with options:
#   curl -sSL ... | bash -s -- --user          # Install for current user only
#   curl -sSL ... | bash -s -- --system        # Install system-wide (requires sudo)
#   curl -sSL ... | bash -s -- --uninstall     # Remove installation
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/Barneyjm/ai-os"
INSTALL_DIR_USER="$HOME/.local/share/meerai"
INSTALL_DIR_SYSTEM="/opt/meerai"
CONFIG_DIR_USER="$HOME/.config/meerai"
CONFIG_DIR_SYSTEM="/etc/meerai"
VENV_NAME="venv"
MIN_PYTHON_VERSION="3.10"

# Parse arguments
INSTALL_MODE="user"
UNINSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            INSTALL_MODE="user"
            shift
            ;;
        --system)
            INSTALL_MODE="system"
            shift
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --help|-h)
            echo "MeerAI Install Script"
            echo ""
            echo "Usage: install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --user       Install for current user only (default)"
            echo "  --system     Install system-wide (requires sudo)"
            echo "  --uninstall  Remove MeerAI installation"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set directories based on mode
if [[ "$INSTALL_MODE" == "system" ]]; then
    INSTALL_DIR="$INSTALL_DIR_SYSTEM"
    CONFIG_DIR="$CONFIG_DIR_SYSTEM"
    BIN_DIR="/usr/local/bin"
    SYSTEMD_DIR="/etc/systemd/system"
    NEED_SUDO=true
else
    INSTALL_DIR="$INSTALL_DIR_USER"
    CONFIG_DIR="$CONFIG_DIR_USER"
    BIN_DIR="$HOME/.local/bin"
    SYSTEMD_DIR="$HOME/.config/systemd/user"
    NEED_SUDO=false
fi

# Helper functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

run_cmd() {
    if [[ "$NEED_SUDO" == true ]]; then
        sudo "$@"
    else
        "$@"
    fi
}

check_python() {
    info "Checking Python version..."

    # Try python3 first, then python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        error "Python not found. Please install Python $MIN_PYTHON_VERSION or later."
    fi

    # Check version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
        error "Python $MIN_PYTHON_VERSION or later required. Found: $PYTHON_VERSION"
    fi

    success "Python $PYTHON_VERSION found"
}

check_dependencies() {
    info "Checking dependencies..."

    # Check for git
    if ! command -v git &> /dev/null; then
        error "git not found. Please install git."
    fi

    # Check for pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        error "pip not found. Please install pip."
    fi

    success "All dependencies found"
}

uninstall() {
    info "Uninstalling MeerAI..."

    # Stop service if running
    if [[ "$INSTALL_MODE" == "system" ]]; then
        if systemctl is-active --quiet meerai-agent 2>/dev/null; then
            info "Stopping meerai-agent service..."
            run_cmd systemctl stop meerai-agent
            run_cmd systemctl disable meerai-agent
        fi
    else
        if systemctl --user is-active --quiet meerai-agent 2>/dev/null; then
            info "Stopping meerai-agent service..."
            systemctl --user stop meerai-agent
            systemctl --user disable meerai-agent
        fi
    fi

    # Remove files
    if [[ -d "$INSTALL_DIR" ]]; then
        info "Removing $INSTALL_DIR..."
        run_cmd rm -rf "$INSTALL_DIR"
    fi

    # Remove symlinks
    for cmd in meerai meerai-agent meerai-shell; do
        if [[ -L "$BIN_DIR/$cmd" ]]; then
            info "Removing $BIN_DIR/$cmd..."
            run_cmd rm -f "$BIN_DIR/$cmd"
        fi
    done

    # Remove systemd unit
    if [[ -f "$SYSTEMD_DIR/meerai-agent.service" ]]; then
        info "Removing systemd unit..."
        run_cmd rm -f "$SYSTEMD_DIR/meerai-agent.service"
        if [[ "$INSTALL_MODE" == "system" ]]; then
            run_cmd systemctl daemon-reload
        else
            systemctl --user daemon-reload
        fi
    fi

    success "MeerAI uninstalled"
    info "Config files in $CONFIG_DIR were preserved. Remove manually if desired."
    exit 0
}

install() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         MeerAI Installer             ║${NC}"
    echo -e "${GREEN}║   Your AI that watches while you rest ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
    echo ""

    check_python
    check_dependencies

    # Check for sudo if system install
    if [[ "$INSTALL_MODE" == "system" ]]; then
        if [[ $EUID -ne 0 ]]; then
            info "System install requires sudo privileges"
            sudo -v || error "sudo access required for system install"
        fi
    fi

    # Create directories
    info "Creating directories..."
    run_cmd mkdir -p "$INSTALL_DIR"
    run_cmd mkdir -p "$CONFIG_DIR"
    run_cmd mkdir -p "$BIN_DIR"
    run_cmd mkdir -p "$SYSTEMD_DIR"

    # Clone or update repo
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        info "Updating existing installation..."
        cd "$INSTALL_DIR"
        run_cmd git pull origin main
    else
        info "Downloading MeerAI..."
        run_cmd git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    # Create virtual environment
    info "Creating Python virtual environment..."
    $PYTHON_CMD -m venv "$INSTALL_DIR/$VENV_NAME"

    # Install dependencies
    info "Installing Python dependencies..."
    "$INSTALL_DIR/$VENV_NAME/bin/pip" install --upgrade pip
    "$INSTALL_DIR/$VENV_NAME/bin/pip" install -e "$INSTALL_DIR[dev]"

    # Create wrapper scripts
    info "Creating command wrappers..."

    cat > "$INSTALL_DIR/bin/meerai-agent" << 'AGENT_WRAPPER'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SCRIPT_DIR/system-agent:$SCRIPT_DIR/ai-shell:$PYTHONPATH"
exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/system-agent/agent.py" "$@"
AGENT_WRAPPER

    cat > "$INSTALL_DIR/bin/meerai-shell" << 'SHELL_WRAPPER'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SCRIPT_DIR/system-agent:$SCRIPT_DIR/ai-shell:$PYTHONPATH"
exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/ai-shell/shell.py" "$@"
SHELL_WRAPPER

    cat > "$INSTALL_DIR/bin/meerai" << 'MEERAI_WRAPPER'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default to shell if no args, otherwise pass to agent
if [[ $# -eq 0 ]]; then
    exec "$SCRIPT_DIR/bin/meerai-shell"
else
    exec "$SCRIPT_DIR/bin/meerai-agent" "$@"
fi
MEERAI_WRAPPER

    run_cmd chmod +x "$INSTALL_DIR/bin/meerai-agent"
    run_cmd chmod +x "$INSTALL_DIR/bin/meerai-shell"
    run_cmd chmod +x "$INSTALL_DIR/bin/meerai"
    run_cmd mkdir -p "$INSTALL_DIR/bin"

    # Create symlinks
    info "Creating command symlinks..."
    run_cmd ln -sf "$INSTALL_DIR/bin/meerai" "$BIN_DIR/meerai"
    run_cmd ln -sf "$INSTALL_DIR/bin/meerai-agent" "$BIN_DIR/meerai-agent"
    run_cmd ln -sf "$INSTALL_DIR/bin/meerai-shell" "$BIN_DIR/meerai-shell"

    # Copy default config if not exists
    if [[ ! -f "$CONFIG_DIR/agency.toml" ]]; then
        info "Installing default configuration..."
        run_cmd cp "$INSTALL_DIR/config/agency.toml" "$CONFIG_DIR/agency.toml"
    fi

    # Install systemd unit
    info "Installing systemd service..."
    if [[ "$INSTALL_MODE" == "system" ]]; then
        run_cmd cp "$INSTALL_DIR/services/meerai-agent.service" "$SYSTEMD_DIR/"
        run_cmd systemctl daemon-reload
    else
        cp "$INSTALL_DIR/services/meerai-agent-user.service" "$SYSTEMD_DIR/meerai-agent.service"
        systemctl --user daemon-reload
    fi

    echo ""
    success "MeerAI installed successfully!"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo ""
    echo "  1. Set your API key:"
    echo -e "     ${YELLOW}export ANTHROPIC_API_KEY='your-key-here'${NC}"
    echo "     (or OPENAI_API_KEY for OpenAI/Fireworks)"
    echo ""
    echo "  2. Start the shell:"
    echo -e "     ${YELLOW}meerai${NC}"
    echo ""
    echo "  3. Or run as a daemon:"
    if [[ "$INSTALL_MODE" == "system" ]]; then
        echo -e "     ${YELLOW}sudo systemctl enable --now meerai-agent${NC}"
    else
        echo -e "     ${YELLOW}systemctl --user enable --now meerai-agent${NC}"
    fi
    echo ""
    echo "  4. Configure triggers:"
    echo -e "     ${YELLOW}$CONFIG_DIR/agency.toml${NC}"
    echo ""

    # Check if BIN_DIR is in PATH
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        warn "$BIN_DIR is not in your PATH"
        echo "  Add it with:"
        echo -e "     ${YELLOW}echo 'export PATH=\"\$PATH:$BIN_DIR\"' >> ~/.bashrc${NC}"
        echo ""
    fi
}

# Main
if [[ "$UNINSTALL" == true ]]; then
    uninstall
else
    install
fi
