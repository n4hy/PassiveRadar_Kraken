#!/bin/bash
set -e

echo "Compiling C++ kernels..."
cd src
make clean
make -j$(nproc)

echo "Copying libraries to Python package..."
cp libkraken_*.so ../gr-kraken_passive_radar/python/kraken_passive_radar/

echo "Libraries updated. Please run './build_oot.sh' or 'sudo make install' in the OOT build directory to reinstall the module."
cd ..
