# Getting Started with meetkatOS

## Quick Development Setup

### 1. Install Dependencies

```bash
# Python packages
pip install aiohttp prompt_toolkit rich

# For Python < 3.11
pip install tomli
```

### 2. Get a Model

```bash
mkdir -p models
cd models

# Small model for testing (~2GB)
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf

# Or better quality (~4.5GB)
# wget https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

### 3. Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build -j$(nproc)
```

### 4. Run Everything

Terminal 1 - Inference Server:
```bash
./llama.cpp/build/bin/llama-server \
    --model models/Phi-3-mini-4k-instruct-q4.gguf \
    --port 8080
```

Terminal 2 - System Agent:
```bash
export AI_RUNTIME_URL="http://127.0.0.1:8080"
export AI_DEV_MODE=1
python system-agent/agent.py
```

Terminal 3 - AI Shell:
```bash
export AI_AGENT_URL="http://127.0.0.1:8000"
python ai-shell/shell.py
```

## Try It Out

```
❯ What files are in my home directory?
❯ Show me system memory usage
❯ /profile list
❯ /profile set focus
❯ Create a file called test.txt with hello world
```

## Environment Variables

| Variable           | Description                       | Default                  |
|--------------------|-----------------------------------|--------------------------|
| AI_RUNTIME_URL     | HTTP URL for inference server     | (uses socket)            |
| AI_RUNTIME_SOCKET  | Unix socket for inference         | /run/ai-runtime.sock     |
| AI_AGENT_URL       | HTTP URL for agent (dev mode)     | (uses socket)            |
| AI_AGENT_SOCKET    | Unix socket for agent             | /run/system-agent.sock   |
| AI_DEV_MODE        | Enable HTTP server mode           | (disabled)               |
| AI_POLICY_CONFIG   | Path to agency.toml               | (auto-detect)            |
