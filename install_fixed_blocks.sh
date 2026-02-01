#!/bin/bash
# Copy corrected GRC block definitions to installation directory
sudo cp gr-kraken_passive_radar/grc/*.block.yml /usr/local/share/gnuradio/grc/blocks/
echo "Block files installed successfully"
# Clear GRC cache to force reload
rm -rf ~/.cache/grc_gnuradio 2>/dev/null
echo "GRC cache cleared"
echo "Please restart GNU Radio Companion to see the changes"
