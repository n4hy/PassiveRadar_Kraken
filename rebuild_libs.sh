#!/bin/bash
set -eu  # Exit on error, treat unset variables as errors

# Get script directory for reliable paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Compiling C++ kernels (out-of-source build)..."
cd "$SCRIPT_DIR/src" || { echo "ERROR: Cannot enter src directory"; exit 1; }
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
cd "$SCRIPT_DIR/src"

echo "Copying libraries to Python package..."
# Copy to local source tree
cp -f libkraken_*.so "$SCRIPT_DIR/gr-kraken_passive_radar/python/kraken_passive_radar/" || {
    echo "Warning: Could not copy to local source tree"
}

# Dynamically detect Python installation directory
INSTALL_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'] + '/kraken_passive_radar')" 2>/dev/null || echo "")
if [ -z "$INSTALL_DIR" ]; then
    echo "Warning: Could not detect Python installation path"
    # Fallback to common location
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3")
    INSTALL_DIR="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/kraken_passive_radar"
fi
echo "Python install directory: $INSTALL_DIR"

# Try to copy to installed location if writable (dev helper)
if [ -d "$INSTALL_DIR" ]; then
    echo "Attempting to copy libraries to install dir: $INSTALL_DIR"
    if [ "$EUID" -eq 0 ]; then
        cp -f libkraken_*.so "$INSTALL_DIR/" || echo "Warning: Could not copy to install dir"
    else
        sudo cp -f libkraken_*.so "$INSTALL_DIR/" || echo "Warning: Could not copy to install dir (permission denied?)"
    fi

    echo "Verifying installation:"
    ls -l "$INSTALL_DIR/libkraken_eca_b_clutter_canceller.so" 2>/dev/null || echo "Library not found in install dir"

    echo "Checking dependencies of installed library:"
    ldd "$INSTALL_DIR/libkraken_eca_b_clutter_canceller.so" 2>/dev/null || true
else
    echo "Install directory not found: $INSTALL_DIR"
    echo "This is normal if the module hasn't been installed yet."
fi

echo ""
echo "Libraries updated. Please run './build_oot.sh' or 'sudo make install' in the OOT build directory to reinstall the module."
cd "$SCRIPT_DIR"
