#!/bin/bash
set -eu  # Exit on error, treat unset variables as errors

# Copy corrected GRC block definitions to installation directory
# Dynamically detect GRC blocks directory
GRC_BLOCKS_DIR=$(python3 -c "import os; import gnuradio; gr_prefix = os.path.dirname(os.path.dirname(gnuradio.__file__)); print(os.path.join(gr_prefix, 'share/gnuradio/grc/blocks'))" 2>/dev/null || echo "/usr/local/share/gnuradio/grc/blocks")

echo "Installing GRC blocks to: $GRC_BLOCKS_DIR"
sudo cp gr-kraken_passive_radar/grc/*.block.yml "$GRC_BLOCKS_DIR/"
echo "Block files installed successfully"

# Clear GRC cache to force reload
rm -rf ~/.cache/grc_gnuradio 2>/dev/null || true
echo "GRC cache cleared"
echo "Please restart GNU Radio Companion to see the changes"
