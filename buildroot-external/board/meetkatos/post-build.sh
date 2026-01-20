#!/bin/bash
set -e
TARGET_DIR="$1"

mkdir -p "${TARGET_DIR}/models/system"
mkdir -p "${TARGET_DIR}/var/meetkatos"
mkdir -p "${TARGET_DIR}/etc/meetkatos"

echo "meetkatos" > "${TARGET_DIR}/etc/hostname"

cat > "${TARGET_DIR}/etc/motd" << 'EOF'
                     _   _         _    ___  ____
 _ __ ___   ___  ___| |_| | ____ _| |_ / _ \/ ___|
| '_ ` _ \ / _ \/ _ \ __| |/ / _` | __| | | \___ \
| | | | | |  __/  __/ |_|   < (_| | |_| |_| |___) |
|_| |_| |_|\___|\___|\__|_|\_\__,_|\__|\___/|____/

Welcome to meetkatOS
EOF
