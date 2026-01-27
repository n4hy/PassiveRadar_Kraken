#!/bin/bash
# Helper script to build the OOT module
set -e

# Get script directory for reliable paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building OOT module..."
cd "$SCRIPT_DIR/gr-kraken_passive_radar" || { echo "ERROR: Cannot enter gr-kraken_passive_radar directory"; exit 1; }
rm -rf build
mkdir -p build || { echo "ERROR: Cannot create build directory"; exit 1; }
cd build || { echo "ERROR: Cannot enter build directory"; exit 1; }

# Detect Python installation directory
PY_SITE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null)
if [ -z "$PY_SITE" ]; then
    echo "ERROR: Could not detect Python installation directory"
    exit 1
fi
echo "Detected Python install dir: $PY_SITE"

# Configure with explicit Python path and force RELEASE build type with architecture optimizations
# User preferred install prefix: /usr/local
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DGR_PYTHON_DIR="$PY_SITE" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" ..

# Build
make -j"$(nproc)"

echo ""
echo "Build successful."
echo "Now running: sudo make install && sudo ldconfig"

# Handle sudo appropriately
if [ "$EUID" -eq 0 ]; then
    make install && ldconfig
else
    sudo make install && sudo ldconfig
fi

cd "$SCRIPT_DIR"
echo "OOT module build and install complete."
