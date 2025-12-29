#!/bin/bash
set -e

echo "Compiling C++ kernels..."
cd src
make clean
make -j$(nproc)

echo "Copying libraries to Python package..."
# Copy to local source tree
cp -f libkraken_*.so ../gr-kraken_passive_radar/python/kraken_passive_radar/

# Try to copy to installed location if writable (dev helper)
INSTALL_DIR="/usr/local/lib/python3.12/dist-packages/kraken_passive_radar"
if [ -d "$INSTALL_DIR" ]; then
    echo "Attempting to copy libraries to install dir: $INSTALL_DIR"
    sudo cp -f libkraken_*.so "$INSTALL_DIR/" || echo "Warning: Could not copy to install dir (permission denied?)"

    echo "Verifying installation:"
    ls -l "$INSTALL_DIR/libkraken_eca_b_clutter_canceller.so"

    echo "Checking dependencies of installed library:"
    ldd "$INSTALL_DIR/libkraken_eca_b_clutter_canceller.so" || true
else
    echo "Install directory not found: $INSTALL_DIR"
fi

echo "Libraries updated. Please run './build_oot.sh' or 'sudo make install' in the OOT build directory to reinstall the module."
cd ..
