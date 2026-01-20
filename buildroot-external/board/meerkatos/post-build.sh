#!/bin/bash
set -e
TARGET_DIR="$1"

mkdir -p "${TARGET_DIR}/models/system"
mkdir -p "${TARGET_DIR}/var/meerkatos"
mkdir -p "${TARGET_DIR}/etc/meerkatos"

echo "meerkatos" > "${TARGET_DIR}/etc/hostname"

cat > "${TARGET_DIR}/etc/motd" << 'EOF'
  __  __                _         _    ___  ____
 |  \/  | ___  ___ _ __| | ____ _| |_ / _ \/ ___|
 | |\/| |/ _ \/ _ \ '__| |/ / _` | __| | | \___ \
 | |  | |  __/  __/ |  |   < (_| | |_| |_| |___) |
 |_|  |_|\___|\___|_|  |_|\_\__,_|\__|\___/|____/

Welcome to MeerkatOS
EOF
