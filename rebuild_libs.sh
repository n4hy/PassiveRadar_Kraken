#!/bin/bash
set -e

# Get script directory for reliable paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Compiling C++ kernels..."
cd "$SCRIPT_DIR/src" || { echo "ERROR: Cannot enter src directory"; exit 1; }
make clean || true  # Don't fail if nothing to clean
make -j"$(nproc)"

echo "Copying libraries to Python package..."
# Copy to local source tree
cp -f libkraken_*.so "$SCRIPT_DIR/gr-kraken_passive_radar/python/kraken_passive_radar/" || {
    echo "Warning: Could not copy to local source tree"
}

# Dynamically detect Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
if [ -z "$PYTHON_VERSION" ]; then
    echo "Warning: Could not detect Python version"
    PYTHON_VERSION="3"
fi
echo "Detected Python version: $PYTHON_VERSION"

# Try to copy to installed location if writable (dev helper)
INSTALL_DIR="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/kraken_passive_radar"
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
