#!/bin/bash
# Build and install the gr-kraken_passive_radar OOT module
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OOT_DIR="$SCRIPT_DIR/gr-kraken_passive_radar"
BUILD_DIR="$OOT_DIR/build"

echo "=== Building gr-kraken_passive_radar OOT module ==="

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
cmake "$OOT_DIR" -DCMAKE_BUILD_TYPE=Release

# Build (use all cores)
make -j"$(nproc)"

# Install (requires sudo for system-wide install)
echo "=== Installing OOT module ==="
sudo make install

# Update linker cache
sudo ldconfig

echo "=== OOT module built and installed successfully ==="
