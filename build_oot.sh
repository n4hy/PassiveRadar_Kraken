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

# Use system Python (where gnuradio is installed) instead of pyenv
# gnuradio Ubuntu packages install for the system Python only
SYSTEM_PYTHON="/usr/bin/python3"
if [ ! -x "$SYSTEM_PYTHON" ]; then
    echo "ERROR: System python3 not found at $SYSTEM_PYTHON"
    exit 1
fi

PY_SITE=$($SYSTEM_PYTHON -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null)
if [ -z "$PY_SITE" ]; then
    echo "ERROR: Could not detect Python installation directory"
    exit 1
fi
echo "Detected Python install dir: $PY_SITE"

# Eigen3 may be installed to ~/.local (user-level install)
EXTRA_CMAKE_ARGS=""
if [ -d "$HOME/.local/share/eigen3/cmake" ]; then
    EXTRA_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$HOME/.local"
    echo "Using Eigen3 from $HOME/.local"
fi

# Configure with explicit Python path and force RELEASE build type
# Architecture-specific optimization is handled by each module's CMakeLists.txt
# Use -DNATIVE_OPTIMIZATION=ON if building and running on the same machine
cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DGR_PYTHON_DIR="$PY_SITE" \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE="$SYSTEM_PYTHON" \
      $EXTRA_CMAKE_ARGS \
      -Wno-dev \
      ..

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
