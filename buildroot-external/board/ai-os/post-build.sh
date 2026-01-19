#!/bin/bash
set -e
TARGET_DIR="$1"

mkdir -p "${TARGET_DIR}/models/system"
mkdir -p "${TARGET_DIR}/var/ai-os"
mkdir -p "${TARGET_DIR}/etc/ai-os"

echo "ai-os" > "${TARGET_DIR}/etc/hostname"

cat > "${TARGET_DIR}/etc/motd" << 'EOF'
    _    ___      ___  ____
   / \  |_ _|    / _ \/ ___|
  / _ \  | |____| | | \___ \
 / ___ \ | |____| |_| |___) |
/_/   \_\___|    \___/|____/

Welcome to AI-OS
EOF
