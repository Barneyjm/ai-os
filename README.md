# AI-OS: An AI-First Operating System

An experimental operating system where AI is the primary interface, not an afterthought.

## Vision

Traditional operating systems bolt AI on top as another app. AI-OS inverts this: the AI agent is a core system component with deep awareness of and control over the machine. You express intent, the system figures out the rest.

## Features

- **Conversational Interface**: Natural language as the primary interaction mode
- **Agency Policy System**: User-defined autonomy levels per domain (filesystem, processes, packages, etc.)
- **Profile Switching**: Instant context switching (focus, gaming, away, cautious modes)
- **Adaptive Learning**: System learns your preferences over time
- **Policy Transparency**: Always know why the AI did or didn't do something

## Quick Start

```bash
# Install dependencies
pip install aiohttp prompt_toolkit rich

# Download a model (Phi-3 mini for testing)
mkdir -p models
cd models
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
cd ..

# Build and run llama.cpp server
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build -j
./build/bin/llama-server --model ../models/Phi-3-mini-4k-instruct-q4.gguf --port 8080

# In another terminal, run the agent (dev mode)
export AI_RUNTIME_URL="http://127.0.0.1:8080"
python system-agent/agent.py

# In a third terminal, run the shell
python ai-shell/shell.py
```

## Shell Commands

| Command             | Description                 |
|---------------------|-----------------------------|
| `/help`             | Show available commands     |
| `/profile`          | Show current agency profile |
| `/profile list`     | List all profiles           |
| `/profile set NAME` | Switch to a profile         |
| `/policy DOMAIN OP` | Check policy for an action  |
| `/reset`            | Reset conversation          |
| `/exit`             | Exit shell                  |

## Agency Profiles

- **default**: Balanced autonomy for daily use
- **cautious**: Maximum confirmation, good for learning or sensitive work
- **focus**: Minimize interruptions, handle routine tasks silently
- **away**: Full autonomy while you're gone
- **gaming**: Minimal AI activity, maximum resources to apps
- **presentation**: No interruptions, stability mode

## Project Structure

```
ai-os/
├── config/agency.toml       # Agency policy configuration
├── system-agent/
│   ├── agent.py             # Core AI agent
│   └── policy.py            # Policy engine
├── ai-shell/
│   └── shell.py             # Terminal interface
├── services/                # runit service definitions
├── buildroot-external/      # Linux distro build config
└── docs/
    ├── ARCHITECTURE.md
    └── GETTING_STARTED.md
```

## License

MIT

## Contributing

This is an experimental project exploring AI-OS paradigms. Ideas, experiments, and PRs welcome.
