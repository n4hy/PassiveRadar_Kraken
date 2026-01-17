#!/bin/bash
# Wrapper script to run unit tests reliably
# Ensures the tests directory is targeted and OOT python path is handled by the tests themselves.

# Check for compiled libraries
if [ ! -f "gr-kraken_passive_radar/python/kraken_passive_radar/libkraken_doppler_processing.so" ]; then
    echo "WARNING: C++ Libraries not found in python package directory."
    echo "       Please run './build_oot.sh' to compile them."
    echo "       Note: 'libfftw3-dev' is required for Doppler/CAF processing."
    echo ""
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/gr-kraken_passive_radar/python
python3 -m unittest discover -s tests -v
