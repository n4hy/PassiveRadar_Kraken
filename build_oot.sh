#!/bin/bash
# Helper script to build the OOT module
set -e

echo "Building OOT module..."
cd gr-kraken_passive_radar
rm -rf build
mkdir build
cd build

# Configure
cmake -DCMAKE_INSTALL_PREFIX=/usr ..

# Build
make -j"$(nproc)"

echo ""
echo "Build successful."
echo "Now run: sudo make install && sudo ldconfig"
