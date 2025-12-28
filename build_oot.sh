#!/bin/bash
# Helper script to build the OOT module
set -e

echo "Building OOT module..."
cd gr-kraken_passive_radar
rm -rf build
mkdir build
cd build

# Detect Python installation directory
PY_SITE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "Detected Python install dir: $PY_SITE"

# Configure with explicit Python path
cmake -DCMAKE_INSTALL_PREFIX=/usr -DGR_PYTHON_DIR="$PY_SITE" ..

# Build
make -j"$(nproc)"

echo ""
echo "Build successful."
echo "Now running: sudo make install && sudo ldconfig"
sudo make install && sudo ldconfig
cd -

