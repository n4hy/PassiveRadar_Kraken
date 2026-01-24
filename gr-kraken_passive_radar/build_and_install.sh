#!/bin/bash
# Build and install script for gr-kraken_passive_radar OOT module
set -e

echo "=========================================="
echo "Building gr-kraken_passive_radar OOT Module"
echo "=========================================="

# Clean previous build
rm -rf build
mkdir build
cd build

# Configure
echo ""
echo "[1/4] Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native" \
      ..

# Build
echo ""
echo "[2/4] Building..."
make -j$(nproc)

# Install
echo ""
echo "[3/4] Installing (requires sudo)..."
sudo make install

# Update library cache
echo ""
echo "[4/4] Updating library cache..."
sudo ldconfig

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
echo ""

# Verification checks
echo "[Check 1] GRC block YAML files:"
ls -la /usr/local/share/gnuradio/grc/blocks/*kraken* 2>/dev/null || \
ls -la /usr/share/gnuradio/grc/blocks/*kraken* 2>/dev/null || \
echo "  WARNING: No YAML files found in standard paths"

echo ""
echo "[Check 2] Shared library:"
ldconfig -p | grep kraken || echo "  WARNING: Library not in ldconfig"

echo ""
echo "[Check 3] Python module import test:"
python3 -c "from gnuradio import kraken_passive_radar; print('  SUCCESS: Module imported'); print('  Blocks:', dir(kraken_passive_radar))" 2>&1 || echo "  FAILED: Module import error"

echo ""
echo "=========================================="
echo "If all checks passed, restart GRC:"
echo "  gnuradio-companion"
echo ""
echo "The block should appear under: [Kraken Passive Radar]"
echo "=========================================="
